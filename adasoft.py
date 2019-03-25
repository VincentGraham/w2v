import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class FairseqCriterion(_Loss):
    def __init__(self, pad_idx):
        super().__init__()
        self.padding_idx = pad_idx

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with two elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)


class FacebookAdaptiveLoss(FairseqCriterion):
    """This is an implementation of the loss function accompanying the adaptive softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, task):
        super().__init__(task)

    def forward(self, asm, input, target, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with two elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        """

        adaptive_softmax = asm

        # net_output = model(**sample['net_input'])
        # target = model.get_targets(sample, net_output).view(-1)
        net_output = input
        target = target.contiguous().view(-1)

        bsz = target.size(0)

        logits, target = adaptive_softmax(net_output[0], target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert (target[i].min() >= 0
                        and target[i].max() <= logits[i].size(1))
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    size_average=False,
                    ignore_index=self.padding_idx,
                    reduce=reduce)

        sample_size = target.size(0)

        return loss, sample_size


class FacebookAdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim

        self.lsm = nn.LogSoftmax(dim=1)
        self.head = nn.Linear(input_dim, output_dim, bias=False)
        self._make_tail(True)

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('version', torch.LongTensor([1]))
        # versions prior to 1 had a bug that offset indices on the head by 1
        self.buggy_offset = 0

    def _make_tail(self, fix_exponent):
        extra_denom = 1 if fix_exponent else 0

        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            self.tail.append(
                nn.Sequential(
                    nn.Linear(
                        self.input_dim,
                        self.input_dim // 4**(i + extra_denom),
                        bias=False), nn.Dropout(self.dropout),
                    nn.Linear(
                        self.input_dim // 4**(i + extra_denom),
                        self.cutoff[i + 1] - self.cutoff[i],
                        bias=False)))

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            self.buggy_offset = 1
            self._make_tail(False)
            state_dict[version_name] = torch.LongTensor([1])

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.contiguous().view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i - self.buggy_offset

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        # input = input.contiguous().view(-1, input.size(-1))
        input = F.dropout(input, p=self.dropout, training=self.training)

        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                print(target_idxs[i])
                output.append(self.tail[i](input.index_select(
                    0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0] - self.buggy_offset:head_sz -
                                self.buggy_offset].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(
                    tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(
                    tail_priors[idxs, i, None])

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs


class AdaptiveSoftmax(nn.Module):
    """Adaptive Softmax output layer

    Args:
        input_size: size of each input sample
        cutoff: indexes of words that splited into each bucket
        reduce_factor: dimension reduction factor of each tail bucket before projected
            to each words. Default: 4

    Shape:
        - input: (N, input_size)
        - target: (N)
        - output: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]

    Attributes:
        head: the learnable weights of the module for head bucket
        tail: the learnable weights of the module for tail buckets

    Examples::

        >>> m = AdaptiveSoftmax(20, [2000, 10000])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> log_prob = m.log_prob(input)
    """

    def __init__(self, input_size, cutoff, reduce_factor=4):
        super().__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()

        for i in range(len(cutoff) - 1):
            if reduce_factor == 1:
                seq = nn.Linear(input_size, cutoff[i + 1] - cutoff[i])

            else:
                seq = nn.Sequential(
                    nn.Linear(input_size, input_size // reduce_factor**i,
                              False),
                    nn.Linear(input_size // reduce_factor**i,
                              cutoff[i + 1] - cutoff[i]),
                )

            self.tail.append(seq)

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.any():
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target=None):
        output = [self.head(input)]

        if target is not None:
            self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                print(self.id[i])
                output.append(self.tail[i](input.index_select(0, self.id[i])))

            else:
                output.append(None)

        return output

    def log_prob(self, input):
        with torch.no_grad():
            head_out = self.head(input)

            batch_size = head_out.size(0)
            prob = torch.empty(
                batch_size, self.cutoff[-1], device=input.device)

            lsm_head = F.log_softmax(head_out, 1)
            prob[:, :self.cutoff[0]].copy_(lsm_head[:, :self.cutoff[0]])

            for i in range(len(self.tail)):
                split = lsm_head[:, self.cutoff[0] + i].unsqueeze(1)
                lsm_tail = F.log_softmax(self.tail[i](input), 1)
                prob[:, self.cutoff[i]:self.cutoff[i + 1]].copy_(
                    lsm_tail).add_(split)

        return prob


class TiedAdaptiveSoftmax(nn.Module):
    """Adaptive Softmax that supports weight tying

    Args:
        weight: weight tensor for each words of shape [num_words, dim]
        cutoff: indexes of words that splited into each bucket

    Shape:
        - input: (N, input_size)
        - output: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]

    Attributes:
        weight: the learnable weights of the module that tied with specified tensor
        biases: the learnable biases of the module

    Examples::

        >>> m = TiedAdaptiveSoftmax(20, [2000, 10000])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> log_prob = m.log_prob(input)
    """

    def __init__(self, weight, cutoff):
        super().__init__()

        self.weight = weight
        self.biases = nn.ParameterList()
        self.biases.append(nn.Parameter(torch.zeros(cutoff[0])))
        for i in range(len(cutoff) - 1):
            self.biases.append(
                nn.Parameter(torch.zeros(cutoff[i + 1] - cutoff[i])))

        self.split = nn.Linear(weight.shape[1], len(cutoff) - 1)
        self.cutoff = cutoff

    def set_target(self, target):
        self.id = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))

            if mask.any():
                self.id.append(mask.float().nonzero().squeeze(1))

            else:
                self.id.append(None)

    def forward(self, input, target=None):
        head = F.linear(input, self.weight[:self.cutoff[0]], self.biases[0])
        split = self.split(input)
        output = [torch.cat([head, split], 1)]

        if target is not None:
            self.set_target(target)

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(
                    F.linear(
                        input.index_select(0, self.id[i]),
                        self.weight[self.cutoff[i]:self.cutoff[i + 1]],
                        self.biases[i + 1],
                    ))
            else:
                output.append(None)

        return output

    def log_prob(self, input):
        with torch.no_grad():
            linear_out = F.linear(input, self.weight,
                                  torch.cat([p for p in self.biases]))
            split = self.split(input)
            head = F.log_softmax(
                torch.cat([linear_out[:, :self.cutoff[0]], split], 1), 1)
            linear_out[:, :self.cutoff[0]].copy_(head[:, :-split.shape[1]])

            for i in range(len(self.cutoff) - 1):
                part = linear_out[:, self.cutoff[i]:self.cutoff[i + 1]]
                part.copy_(F.log_softmax(part, 1))
                part.add_(head[:, self.cutoff[0] + i].unsqueeze(1))

        return linear_out


class AdaptiveLoss(nn.Module):
    """Loss layer for Adaptive Softmax

    Args:
        cutoff: indexes of words that splited into each bucket

    Shape:
        - input: [(N, cutoff[0] + len(cutoff) - 1), (N_1, cutoff[1] - cutoff[0]), ...]
        - target: (N)

    Examples::

        >>> cutoff = [2000, 10000]
        >>> m = AdaptiveSoftmax(20, cutoff)
        >>> criterion = AdaptiveLoss(cutoff)
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(low=0, high=10000, size=[128])
        >>> output = m(input, target)
        >>> loss = criterion(output, target)
        >>> loss.backward()
    """

    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def remap_target(self, target):
        new_target = [target.clone()]

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                new_target.append(target[mask].add(-self.cutoff[i]))

            else:
                new_target.append(None)

        return new_target

    def forward(self, input, target):
        batch_size = input[0].size(0)
        target = self.remap_target(target.data)

        output = 0.0

        for i in range(len(input)):
            if input[i] is not None:
                print(target[i].min(), target[i].max(), input[i].size(1))
                assert target[i].min() >= 0 and target[i].max(
                ) <= input[i].size(1)
                output = output + F.cross_entropy(
                    input[i], target[i], size_average=False)

        output /= batch_size

        return output
