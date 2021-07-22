import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import torch

import utils

T = TypeVar("T")


def read_data(file):
    with open(file) as f:
        docs = json.load(f)
    return docs


def _apply(s: str, f: Optional[Callable[[str], str]]) -> str:
    return f(s) if f is not None else s


class Preprocessor:
    serialize_embeddings: bool = False

    def __init__(self):
        self.vocabs: Dict[str, utils.data.Vocab] = {}
        self._embeddings: Optional[torch.Tensor] = None
        self._embed_size: Optional[int] = None
        self._embed_file: Optional[Union[str, bytes, os.PathLike]] = None

    def build_vocab(
        self,
        file: Union[str, bytes, os.PathLike],
        unknown: str = "<UNK>",
        preprocess: Optional[Callable[[str], str]] = str.lower,
        cache_dir: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> None:
        def _build_vocabs(file):
            word_vocab = utils.data.Vocab(unknown)
            word_vocab.preprocess = preprocess
            char_vocab = utils.data.Vocab(unknown)
            label_vocab = utils.data.Vocab()

            for doc in read_data(file):
                for token in doc["forms"]:
                    word_vocab(token)
                    for char in token:
                        char_vocab(char)
                for span in doc["entities"]:
                    label_vocab(span["label"])
            word_vocab.freeze()
            char_vocab.freeze()
            label_vocab.freeze()
            return {"word": word_vocab, "char": char_vocab, "label": label_vocab}

        self.vocabs.update(_wrap_cache(_build_vocabs, file, cache_dir, suffix=".vocab"))

    def load_embeddings(
        self,
        file: Union[str, bytes, os.PathLike],
        unknown: str = "<UNK>",
        preprocess: Optional[Callable[[str], str]] = str.lower,
        cache_dir: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> None:
        def _add_entry(token):
            if token not in vocab:
                nonlocal embeddings
                vocab.append(token)
                embeddings = torch.vstack((embeddings, torch.zeros_like(embeddings[0])))

        vocab, embeddings = _wrap_cache(self._load_embeddings, file, cache_dir)
        _add_entry(unknown)
        self.vocabs["word"] = utils.data.Vocab.fromkeys(vocab, unknown)
        self.vocabs["word"].preprocess = preprocess
        self._embeddings = embeddings
        self._embed_size = embeddings.size(1)
        self._embed_file = file

    @staticmethod
    def _load_embeddings(file) -> Tuple[List[str], torch.Tensor]:
        embeddings = utils.data.load_embeddings(file)
        return (list(embeddings.keys()), torch.tensor(list(embeddings.values())))

    def transform(
        self, doc: Tuple[List[str], List[Tuple[int, int, str]]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[int, int, int]]]:
        words, mentions = doc
        sample = (
            torch.tensor([self.vocabs["word"][w] for w in words]),
            [torch.tensor([self.vocabs["char"][c] for c in w]) for w in words],
            [(s[0], s[1], self.vocabs["label"][s[2]]) for s in mentions],
        )
        return sample

    def collate(
        self,
        batch: Iterable[Tuple[torch.Tensor, List[torch.Tensor], List[Tuple[int, int, int]]]],
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[List[Tuple[int, int, int]]]]:
        word_ids, char_ids, mentions = [list(field) for field in zip(*batch)]
        if device is not None:
            word_ids = [ids.to(device) for ids in word_ids]
            char_ids = [[ids.to(device) for ids in seq] for seq in char_ids]
        return word_ids, char_ids, mentions

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if not self.serialize_embeddings:
            state["_embeddings"] = None
        return state

    @property
    def word_embeddings(self) -> Optional[torch.Tensor]:
        if self._embeddings is None and self._embed_file is not None:
            v = self.vocabs["word"]
            assert v.unknown_id is not None
            self.load_embeddings(self._embed_file, v.lookup(v.unknown_id), v.preprocess)
            assert len(self.vocabs["word"]) == len(v)
        return self._embeddings

    @property
    def word_embeddings_dim(self) -> Optional[int]:
        return self._embed_size


def _wrap_cache(load_fn, file, cache_dir=None, suffix=".cache"):
    if cache_dir is None:
        return load_fn(file)

    basename = os.path.basename(file)
    if not basename:
        raise ValueError(f"Invalid filename: '{file}'")
    cache_file = os.path.join(cache_dir, f"{basename}{suffix}")

    if os.path.exists(cache_file):
        obj = torch.load(cache_file)
    else:
        obj = load_fn(file)
        torch.save(obj, cache_file)
    return obj


def create_dataloader(
    file: Union[str, bytes, os.PathLike],
    preprocessor: Preprocessor,
    device: Optional[torch.device] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    def _convert(doc):
        return doc["forms"], [(s["start"], s["end"], s["label"]) for s in doc["entities"]]

    dataset = ListDataset(map(preprocessor.transform, map(_convert, read_data(file))))
    kwargs.setdefault("collate_fn", lambda batch: preprocessor.collate(batch, device))
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader


class ListDataset(list, torch.utils.data.Dataset):
    pass
