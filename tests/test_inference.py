import pytest

torch = pytest.importorskip("torch")

from speech.config import FeatureExtractionConfig, InferenceConfig
from speech.inference import SpeechRecognizer


class DummyCTCModel(torch.nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(feature_dim, num_classes)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            if num_classes > 1:
                self.proj.bias[1] = 1.0

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        logits = self.proj(x)
        return logits, lengths


def test_transcribe_batch_mono_waveform() -> None:
    feature_config = FeatureExtractionConfig()
    inference_config = InferenceConfig(blank_index=0)
    labels = ["_", "a"]
    model = DummyCTCModel(feature_config.n_mels, len(labels))
    recognizer = SpeechRecognizer(
        model,
        feature_config,
        inference_config,
        labels,
        device=torch.device("cpu"),
    )

    waveform = torch.randn(feature_config.sample_rate // 10)

    results = recognizer.transcribe_batch(
        [waveform], sample_rate=feature_config.sample_rate
    )

    assert len(results) == 1
    assert results[0].transcript == "a"
