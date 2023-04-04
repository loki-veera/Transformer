"""
Objective:
---------
I should be able to preprocess sentences into tensors for NLP modelling and use
torch.utils.data.DataLoader for training and validating the model.

Q: What is torchtext?
API which has utilities to create datasets that can be iterated easily for creating a 
language translation model.

Tokenize a raw sentence, build vocabulary and numericalize the tokens to tensors.
Spacy provides support for tokenization in languages other than English.

python -m download spacy <lang>
"""

"""
Implementation from torch website
"""
# import torchtext
# import torch
# from torchtext.data.utils import get_tokenizer
# from collections import Counter
# from torchtext.vocab import Vocab
# from torchtext.utils import download_from_url, extract_archive
# import io
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader

# url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
# train_urls = ('train.de.gz', 'train.en.gz')
# val_urls = ('val.de.gz', 'val.en.gz')
# test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

# train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
# val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
# test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# de_tokenizer = get_tokenizer('spacy', language='de')
# en_tokenizer = get_tokenizer('spacy', language='en')


# def build_vocab(filepath, tokenizer):
#     counter = Counter()
#     with io.open(filepath, encoding='utf-8') as f:
#         for string_ in f:
#             counter.update(tokenizer(string_))
#     return Vocab(counter)

# de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
# en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

# def data_process(filepaths):
#     raw_de_iter = iter(io.open(filepaths[0], encoding='utf8'))
#     raw_en_iter = iter(io.open(filepaths[1], encoding='utf8'))
#     data = []
#     for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
#         de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype = torch.long)
#         en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype = torch.long)
#         data.append((de_tensor_, en_tensor_))
#     return data

# train_data = data_process(train_filepaths)
# val_data = data_process(val_filepaths)
# test_data = data_process(test_filepaths)


# BATCH_SIZE = 128
# PAD_IDX = de_vocab['<pad>']
# BOS_IDX = torch.tensor([de_vocab["<bos>"]])
# EOS_IDX = torch.tensor([de_vocab["<eos>"]])

# def generate_batch(data_batch):
#     de_batch, en_batch = [], []
#     for (de_item, en_item) in data_batch:
#         de_batch.append(torch.cat([BOS_IDX, de_item, EOS_IDX], dim=0))
#         en_batch.append(torch.cat([BOS_IDX, en_item, EOS_IDX], dim=0))
#     de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
#     en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
#     return de_batch, en_batch

# train_iter = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle=True, collate_fn=generate_batch) 
# val_iter = DataLoader(val_data, batch_size= BATCH_SIZE, shuffle=True, collate_fn=generate_batch) 
# test_iter = DataLoader(test_data, batch_size= BATCH_SIZE, shuffle=True, collate_fn=generate_batch) 


"""
Implementation from 
https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71
"""
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("spacy", language='en') # When encountering the apostrophies, spacy is the best.

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def get_vocab(train_datapipeline):
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipeline),
                                      specials=['<UNK>', '<PAD>', '<BOS>', '<EOS>'])

    vocab.set_default_index(vocab['<UNK>'])
    return vocab

