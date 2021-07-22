from collections import defaultdict

import torch
from tqdm import tqdm

import utils
from data import read_data


class ExponentialLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer, decay_steps, decay_rate, staircase=False, last_epoch=-1, verbose=False
    ):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, self.scale, last_epoch, verbose)

    def scale(self, step):
        f = int if self.staircase else float
        return self.decay_rate ** f(step / self.decay_steps)


def create_trainer(model, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), kwargs.pop("lr", 0.001))
    scheduler = ExponentialLR(
        optimizer, kwargs.pop("decay_steps", 100), kwargs.pop("decay_rate", 0.999), staircase=True
    )
    kwargs.setdefault("max_grad_norm", 5.0)
    kwargs.setdefault("step", forward)
    # NOTE: `scheduler.step()` is called every iteration in the trainer
    trainer = utils.training.Trainer(model, (optimizer, scheduler), **kwargs)
    trainer.add_metric("precision", "recall", "f1")
    trainer.add_callback(_CalculateF1(), priority=0)
    trainer.add_callback(utils.training.ProgressCallback())
    return trainer


def forward(model, batch):
    word_ids, char_ids, true_mentions = batch
    spans, scores = model(word_ids, char_ids)
    result = model.compute_metrics(spans, scores, true_mentions)
    return result


class _CalculateF1(utils.training.Callback):
    def on_loop_end(self, context, metrics):
        mode = "train" if context.train else "eval"
        p, r = metrics[f"{mode}/precision"], metrics[f"{mode}/recall"]
        metrics[f"{mode}/f1"] = 2 * p * r / (p + r)

    def on_evaluate_end(self, context, metrics):
        self.on_loop_end(context, metrics)


class EvaluateCallback(utils.training.Callback):
    printer = tqdm.write

    def __init__(self, gold_file, label_map, verbose=False):
        self.gold_file = gold_file
        self.label_map = label_map
        self.verbose = verbose
        self.result = {}
        self._outputs = []

    def on_step_end(self, context, output):
        if context.train:
            return
        self._outputs.extend(output["mentions"])

    def on_loop_end(self, context, metrics):
        if context.train:
            return

        label_map = self.label_map
        docs = read_data(self.gold_file)
        assert len(self._outputs) == len(docs)

        count = defaultdict(int)
        for mentions, doc in zip(self._outputs, docs):
            gold = set((s["start"], s["end"], s["label"]) for s in doc["entities"])
            pred = set((m[0], m[1], label_map[m[2]]) for m in mentions)
            count["ALL/TP"] += len(gold & pred)
            count["ALL/FN"] += len(gold - pred)
            count["ALL/FP"] += len(pred - gold)
            for label in label_map.values():
                sub_gold = set(m for m in gold if m[2] == label)
                sub_pred = set(m for m in pred if m[2] == label)
                count[f"{label}/TP"] += len(sub_gold & sub_pred)
                count[f"{label}/FN"] += len(sub_gold - sub_pred)
                count[f"{label}/FP"] += len(sub_pred - sub_gold)
        self.result.update(count)

        for label in ["ALL"] + list(label_map.values()):
            total = count[f"{label}/TP"] + count[f"{label}/FP"]
            precision = count[f"{label}/TP"] / total if total > 0 else float("nan")
            total = count[f"{label}/TP"] + count[f"{label}/FN"]
            recall = count[f"{label}/TP"] / total if total > 0 else float("nan")
            self.result[f"{label}/precision"] = precision
            self.result[f"{label}/recall"] = recall
            self.result[f"{label}/f1"] = 2 * precision * recall / (precision + recall)

        self._outputs.clear()

    def on_evaluate_end(self, context, metrics):
        r = self.result
        for label in ["ALL"] + list(self.label_map.values()):
            line = "{}\tPrecision: {:.4%}, Recall: {:.4%}, F1: {:.4%}".format(
                label, r[f"{label}/precision"], r[f"{label}/recall"], r[f"{label}/f1"]
            )
            if self.verbose:
                line += "\t(TP: {}, FP: {}, FN: {})".format(
                    r[f"{label}/TP"], r[f"{label}/FP"], r[f"{label}/FN"]
                )
            self.printer(line)
