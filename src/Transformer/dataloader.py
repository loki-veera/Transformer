from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import build_vocab_from_iterator


class DatasetTransforms:
    def __init__(self, src_lang, tgt_lang) -> None:
        multi30k.URL[
            "train"
        ] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL[
            "valid"
        ] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.token_transform = {}
        self.vocab_transform = {}
        self.token_transform[src_lang] = get_tokenizer(
            "spacy", language="de_core_news_sm"
        )
        self.token_transform[tgt_lang] = get_tokenizer(
            "spacy", language="en_core_web_sm"
        )
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.text_tranform = {}

    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.src_lang: 0, self.tgt_lang: 1}
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])

    def generate_transforms(self):
        special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

        for ln in [self.src_lang, self.tgt_lang]:
            train_iter = Multi30k(
                split="train", language_pair=(self.src_lang, self.tgt_lang)
            )
            self.vocab_transform[ln] = build_vocab_from_iterator(
                self.yield_tokens(train_iter, ln),
                min_freq=1,
                specials=special_symbols,
                special_first=True,
            )
        for ln in [self.src_lang, self.tgt_lang]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

    def __sequential_tranforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def __tensor_transform(self, token_ids):
        return torch.cat(
            (
                torch.tensor([self.BOS_IDX]),
                torch.tensor(token_ids),
                torch.tensor([self.EOS_IDX]),
            )
        )

    def apply_transforms(self):
        for lang in [self.src_lang, self.tgt_lang]:
            self.text_tranform[lang] = self.__sequential_tranforms(
                self.token_transform[lang],
                self.vocab_transform[lang],
                self.__tensor_transform,
            )

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_tranform[self.src_lang](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_tranform[self.tgt_lang](tgt_sample.rstrip("\n")))
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch.T, tgt_batch.T
