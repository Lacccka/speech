import pytest

torch = pytest.importorskip("torch")

from speech.config import TrainingConfig
from speech.training import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, features, feature_lengths):
        # Return logits and lengths matching expected Trainer interface
        return features, feature_lengths


def test_train_epoch_empty_iterable(tmp_path) -> None:
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CTCLoss()

    config = TrainingConfig(checkpoint_dir=tmp_path)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=[],
        config=config,
        device=torch.device("cpu"),
    )

    result = trainer.train_epoch()

    assert result == 0.0
