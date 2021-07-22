from functools import lru_cache
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(**kwargs) -> "SpanClassifier":
    word_embeddings = (kwargs.get("word_vocab_size", 1), kwargs.get("word_embed_size", 100))
    char_embeddings = (kwargs.get("char_vocab_size", 1), kwargs.get("char_embed_size", 50))
    if kwargs.get("word_embeddings") is not None:
        word_embeddings = kwargs["word_embeddings"]
    encoder = BiLSTMEncoder(
        word_embeddings,
        char_embeddings,
        char_feature_size=kwargs.get("char_feature_size", 50),
        char_kernel_sizes=kwargs.get("char_kernel_sizes", [3, 4, 5]),
        num_layers=kwargs.get("num_lstm_layers", 3),
        hidden_size=kwargs.get("lstm_hidden_size", 200),
        embedding_dropout=kwargs.get("embedding_dropout", 0.5),
        lstm_dropout=kwargs.get("lstm_dropout", 0.4),
    )
    if isinstance(word_embeddings, torch.Tensor):
        encoder.freeze_embedding()
    scorer = BaselineSpanScorer(
        encoder.out_size,
        num_labels=kwargs.get("num_labels", 0) + SpanClassifier.num_additional_labels,
        mlp_units=kwargs.get("mlp_units", 150),
        mlp_dropout=kwargs.get("mlp_dropout", 0.2),
        feature="concat",
    )
    model = SpanClassifier(encoder, scorer)
    return model


def enumerate_spans(n):
    for i in range(n):
        for j in range(i, n):
            yield (i, j)


@lru_cache  # type: ignore
def get_all_spans(n: int) -> torch.Tensor:
    return torch.tensor(list(enumerate_spans(n)), dtype=torch.long)


class SpanClassifier(nn.Module):
    num_additional_labels = 1

    def __init__(self, encoder: "Encoder", scorer: "SpanScorer"):
        super().__init__()
        self.encoder = encoder
        self.scorer = scorer

    def forward(
        self, *input_ids: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hs, lengths = self.encoder(*input_ids)
        spans = list(map(get_all_spans, lengths))
        scores = self.scorer(hs, spans)
        return spans, scores

    @torch.no_grad()
    def decode(
        self,
        spans: Sequence[torch.Tensor],
        scores: Sequence[torch.Tensor],
    ) -> List[List[Tuple[int, int, int]]]:
        num_labels = self.scorer.num_labels
        mentions = []
        for spans_i, scores_i in zip(spans, scores):
            assert len(spans_i) == len(scores_i)
            labels_i = scores_i.argmax(dim=1)
            mentions_i = [
                (s[0], s[1], label)
                for s, label in zip(spans_i.tolist(), labels_i.tolist())
                if label < num_labels - 1
            ]
            mentions.append(mentions_i)
        return mentions

    def compute_metrics(
        self,
        spans: Sequence[torch.Tensor],
        scores: Sequence[torch.Tensor],
        true_mentions: Sequence[Sequence[Tuple[int, int, int]]],
        decode=True,
    ) -> Dict[str, Any]:
        assert len(spans) == len(scores) == len(true_mentions)
        num_labels = self.scorer.num_labels
        true_labels = []
        for spans_i, scores_i, true_mentions_i in zip(spans, scores, true_mentions):
            assert len(spans_i) == len(scores_i)
            span2idx = {(s[0].item(), s[1].item()): idx for idx, s in enumerate(spans_i)}
            labels_i = torch.full((len(spans_i),), fill_value=num_labels - 1)
            for (start, end, label) in true_mentions_i:
                idx = span2idx.get((start, end))
                if idx is not None:
                    labels_i[idx] = label
            true_labels.append(labels_i)

        scores_flatten = torch.cat(scores)
        true_labels_flatten = torch.cat(true_labels).to(scores_flatten.device)
        assert len(scores_flatten) == len(true_labels_flatten)
        loss = F.cross_entropy(scores_flatten, true_labels_flatten)
        accuracy = categorical_accuracy(scores_flatten, true_labels_flatten)
        result = {"loss": loss, "accuracy": accuracy}

        if decode:
            pred_mentions = self.decode(spans, scores)
            tp, fn, fp = 0, 0, 0
            for pred_mentions_i, true_mentions_i in zip(pred_mentions, true_mentions):
                pred, gold = set(pred_mentions_i), set(true_mentions_i)
                tp += len(gold & pred)
                fn += len(gold - pred)
                fp += len(pred - gold)
            result["precision"] = (tp, tp + fp)
            result["recall"] = (tp, tp + fn)
            result["mentions"] = pred_mentions

        return result


@torch.no_grad()
def categorical_accuracy(
    y: torch.Tensor, t: torch.Tensor, ignore_index: Optional[int] = None
) -> Tuple[int, int]:
    pred = y.argmax(dim=1)
    if ignore_index is not None:
        mask = t == ignore_index
        ignore_cnt = mask.sum()
        pred.masked_fill_(mask, ignore_index)
        count = ((pred == t).sum() - ignore_cnt).item()
        total = (t.numel() - ignore_cnt).item()
    else:
        count = (pred == t).sum().item()
        total = t.numel()
    return count, total


class SpanScorer(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

    def forward(
        self, xs: torch.Tensor, spans: Sequence[torch.Tensor]
    ) -> Union[Sequence[torch.Tensor]]:
        raise NotImplementedError


class BaselineSpanScorer(SpanScorer):
    def __init__(
        self,
        input_size: int,
        num_labels: int,
        mlp_units: Union[int, Sequence[int]] = 150,
        mlp_dropout: float = 0.0,
        feature="concat",
    ):
        super().__init__(num_labels)
        input_size *= 2 if feature == "concat" else 1
        self.mlp = MLP(input_size, num_labels, mlp_units, F.relu, mlp_dropout)
        self.feature = feature

    def forward(
        self, xs: torch.Tensor, spans: Sequence[torch.Tensor]
    ) -> Union[Sequence[torch.Tensor]]:
        max_length = xs.size(1)
        xs_flatten = xs.reshape(-1, xs.size(-1))
        spans_flatten = torch.cat([idxs + max_length * i for i, idxs in enumerate(spans)])
        features = self._compute_feature(xs_flatten, spans_flatten)
        scores = self.mlp(features)
        return torch.split(scores, [len(idxs) for idxs in spans])

    def _compute_feature(self, xs, spans):
        if self.feature == "concat":
            return xs[spans.ravel()].view(len(spans), -1)
        elif self.feature == "minus":
            begins, ends = spans.T
            return xs[ends] - xs[begins]
        else:
            raise NotImplementedError


class Encoder(nn.Module):
    def forward(
        self, word_ids: Sequence[torch.Tensor], char_ids: Sequence[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the encoded sequences and their lengths."""
        raise NotImplementedError

    @property
    def out_size(self) -> int:
        raise NotImplementedError


class BiLSTMEncoder(Encoder):
    def __init__(
        self,
        word_embeddings: Union[torch.Tensor, Tuple[int, int]],
        char_embeddings: Union[torch.Tensor, Tuple[int, int]],
        char_feature_size: int = 50,
        char_kernel_sizes: Union[Sequence[int], int] = 3,
        num_layers: int = 3,
        hidden_size: Optional[int] = None,
        embedding_dropout: float = 0.0,
        lstm_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(word_embeddings, tuple):
            size, dim = word_embeddings
            self.word_emb = nn.Embedding(size, dim)
        else:
            self.word_emb = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.char_cnn = CharCNN(char_embeddings, char_feature_size, char_kernel_sizes)

        lstm_in_size = self.word_emb.weight.size(1) + self.char_cnn.out_size
        if hidden_size is None:
            hidden_size = lstm_in_size
        self.bilstm = nn.LSTM(
            lstm_in_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self._hidden_size = hidden_size

    def freeze_embedding(self) -> None:
        self.word_emb.weight.requires_grad = False

    def forward(
        self, word_ids: Sequence[torch.Tensor], char_ids: Sequence[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([x.size(0) for x in word_ids])
        xs_word = self.word_emb(torch.cat(word_ids, dim=0))
        xs_char = self.char_cnn(list(chain.from_iterable(char_ids)))
        xs = self.embedding_dropout(torch.cat((xs_word, xs_char), dim=1))

        if torch.all(lengths == lengths[0]):
            hs, _ = self.bilstm(xs.view(len(lengths), lengths[0], -1))
        else:
            seq = torch.split(xs, tuple(lengths), dim=0)
            seq = nn.utils.rnn.pack_sequence(seq, enforce_sorted=False)
            hs, _ = self.bilstm(seq)
            hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)
        return self.lstm_dropout(hs), lengths

    @property
    def out_size(self) -> int:
        return self._hidden_size * 2


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int],
        units: Optional[Union[int, Sequence[int]]] = None,
        activate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        units = [units] if isinstance(units, int) else units
        if not units and out_features is None:
            raise ValueError("'out_features' or 'units' must be specified")
        layers = []
        for u in units or []:
            layers.append(MLP.Layer(in_features, u, activate, dropout, bias))
            in_features = u
        if out_features is not None:
            layers.append(MLP.Layer(in_features, out_features, None, 0.0, bias))
        super().__init__(*layers)

    class Layer(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            activate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            dropout: float = 0.0,
            bias: bool = True,
        ):
            super().__init__()
            if activate is not None and not callable(activate):
                raise TypeError("activate must be callable: type={}".format(type(activate)))
            self.linear = nn.Linear(in_features, out_features, bias)
            self.activate = activate
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.linear(x)
            if self.activate is not None:
                h = self.activate(h)
            return self.dropout(h)

        def extra_repr(self) -> str:
            return "{}, activate={}, dropout={}".format(
                self.linear.extra_repr(), self.activate, self.dropout.p
            )

        def __repr__(self):
            return "{}.{}({})".format(MLP.__name__, self._get_name(), self.extra_repr())


class CharCNN(nn.Module):
    def __init__(
        self,
        embeddings: Union[torch.Tensor, Tuple[int, int]],
        feature_size: int,
        kernel_sizes: Union[Sequence[int], int],
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        if isinstance(embeddings, tuple):
            size, dim = embeddings
            self.emb = nn.Embedding(size + 1, dim, padding_idx=-1)
        else:
            embeddings = torch.vstack((embeddings, torch.zeros_like(embeddings[0])))
            self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        in_size = self.emb.weight.size(1)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(in_size, feature_size, k, stride=1, padding=0, bias=bias))
        self.out_size = feature_size * len(kernel_sizes)

    def forward(self, char_ids: Sequence[torch.Tensor]) -> torch.Tensor:
        lengths = torch.tensor([ids.size(0) for ids in char_ids])
        input_ids = nn.utils.rnn.pad_sequence(
            char_ids, batch_first=True, padding_value=self.emb.padding_idx
        )
        max_kernel_size = max([c.kernel_size[0] for c in self.convs])
        if input_ids.size(1) < max_kernel_size:
            pad_shape = (input_ids.size(0), max_kernel_size - input_ids.size(1))
            pads = torch.full(pad_shape, self.emb.padding_idx, device=input_ids.device)
            input_ids = torch.hstack((input_ids, pads))
        xs = self.dropout(self.emb(input_ids))
        ys = [self._foward_conv(c, xs, lengths) for c in self.convs]
        return torch.cat(ys, dim=1)

    @staticmethod
    def _foward_conv(conv, xs, lengths):
        # NOTE: padding is applied to words whose length is less than `kernel_size`.
        n, max_length = xs.shape[:2]
        kernel_size = conv.kernel_size[0]
        mask = torch.ones((n, max_length - kernel_size + 1), dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, : max(length - kernel_size + 1, 1)] = 0
        hs = F.relu(conv(xs.transpose(1, 2)).transpose(1, 2))
        hs.masked_fill_(mask.to(hs.device)[..., None], -float("inf"))
        ys = torch.max(hs, dim=1)[0]
        return ys
