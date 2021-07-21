import torch

import utils


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
    trainer.add_metric("precision", "recall")
    trainer.add_callback(utils.training.ProgressCallback())
    return trainer


def forward(model, batch):
    word_ids, char_ids, true_mentions = batch
    spans, scores = model(word_ids, char_ids)
    result = model.compute_metrics(spans, scores, true_mentions)
    return result
