FROM python:3.7
RUN mkdir /app
WORKDIR /app
COPY generate.py generate.py
COPY load_vocab.py load_vocab.py
COPY rearrange.py rearrange.py

RUN pip install gensim
RUN pip install cython
RUN pip install torch
RUN pip install torchvision
RUN pip install torchtext
RUN pip install matplotlib
CMD ["python", "generate.py"]