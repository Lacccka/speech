"""Inference helpers for speech recognition models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .config import FeatureExtractionConfig, InferenceConfig, ModelConfig
from .features import batch_compute_mel_spectrogram, prepare_ctc_inputs
from .logging import get_logger
from .model import model_from_config


LanguageModelScorer = Callable[[str], float]


def _ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    labels: Sequence[str],
    blank: int,
) -> List[str]:
    batch_size = log_probs.size(0)
    decoded: List[str] = []
    for b in range(batch_size):
        probs = log_probs[b, : lengths[b]]
        best = torch.argmax(probs, dim=-1)
        tokens: List[str] = []
        prev = blank
        for index in best.tolist():
            if index == blank:
                prev = blank
                continue
            if index != prev:
                tokens.append(labels[index])
            prev = index
        decoded.append("".join(tokens))
    return decoded


def _ctc_beam_search(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    labels: Sequence[str],
    blank: int,
    *,
    beam_width: int,
    lm_scorer: Optional[LanguageModelScorer],
    lm_weight: float,
) -> List[str]:
    batch_size = log_probs.size(0)
    decoded: List[str] = []

    for b in range(batch_size):
        T = lengths[b].item()
        beam = {"": (0.0, float("-inf"))}  # prefix -> (log_prob_blank, log_prob_non_blank)

        for t in range(T):
            next_beam = {}
            step_log_probs = log_probs[b, t]
            for prefix, (p_b, p_nb) in beam.items():
                # extend with blank
                prob_blank = step_log_probs[blank].item()
                total_blank = torch.logsumexp(
                    torch.tensor([p_b, p_nb], dtype=torch.float32), dim=0
                ).item() + prob_blank
                _update_beam(next_beam, prefix, total_blank, is_blank=True)

                for idx, label in enumerate(labels):
                    if idx == blank:
                        continue
                    prob = step_log_probs[idx].item()
                    new_prefix = prefix + label

                    lm_score = lm_scorer(new_prefix) if lm_scorer else 0.0
                    lm_bonus = lm_weight * lm_score

                    if prefix and prefix[-len(label) :] == label:
                        total = p_nb + prob + lm_bonus
                    else:
                        total = torch.logsumexp(
                            torch.tensor([p_b, p_nb], dtype=torch.float32), dim=0
                        ).item() + prob + lm_bonus

                    _update_beam(next_beam, new_prefix, total, is_blank=False)

            # prune
            sorted_beam = sorted(
                next_beam.items(),
                key=lambda item: torch.logsumexp(
                    torch.tensor(item[1], dtype=torch.float32), dim=0
                ).item(),
                reverse=True,
            )
            beam = {k: v for k, v in sorted_beam[:beam_width]}

        best_prefix, scores = max(
            beam.items(),
            key=lambda item: torch.logsumexp(
                torch.tensor(item[1], dtype=torch.float32), dim=0
            ).item(),
        )
        decoded.append(best_prefix)
    return decoded


def _update_beam(
    beam: dict,
    prefix: str,
    log_prob: float,
    *,
    is_blank: bool,
) -> None:
    blank_log, non_blank_log = beam.get(prefix, (float("-inf"), float("-inf")))
    if is_blank:
        blank_log = torch.logaddexp(torch.tensor(blank_log), torch.tensor(log_prob)).item()
    else:
        non_blank_log = torch.logaddexp(
            torch.tensor(non_blank_log), torch.tensor(log_prob)
        ).item()
    beam[prefix] = (blank_log, non_blank_log)


@dataclass(slots=True)
class TranscriptionResult:
    """Result of an inference call."""

    transcript: str
    log_probs: Optional[torch.Tensor] = None


class SpeechRecognizer:
    """High level interface for batch and streaming inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        feature_config: FeatureExtractionConfig,
        inference_config: InferenceConfig,
        labels: Sequence[str],
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.feature_config = feature_config
        self.inference_config = inference_config
        self.labels = labels
        self.blank = inference_config.blank_index
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(__name__)
        self.model.to(self.device)
        self.model.eval()
        self.logger.debug(
            "Initialized SpeechRecognizer with model %s on %s",
            self.model.__class__.__name__,
            self.device,
        )

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        feature_config: FeatureExtractionConfig,
        inference_config: InferenceConfig,
        labels: Sequence[str],
        *,
        device: Optional[torch.device] = None,
    ) -> "SpeechRecognizer":
        model = model_from_config(model_config, device=device)
        return cls(model, feature_config, inference_config, labels, device=device)

    def _ensure_sample_rate(self, sample_rate: Optional[int]) -> None:
        if sample_rate is not None and sample_rate != self.feature_config.sample_rate:
            raise ValueError(
                f"Expected audio sampled at {self.feature_config.sample_rate} Hz, "
                f"but received {sample_rate} Hz"
            )

    def _decode(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        *,
        use_beam_search: bool,
        beam_width: Optional[int],
        lm_scorer: Optional[LanguageModelScorer],
        lm_weight: Optional[float],
    ) -> List[str]:
        if use_beam_search:
            width = beam_width or self.inference_config.beam_width
            weight = lm_weight if lm_weight is not None else self.inference_config.lm_weight
            return _ctc_beam_search(
                log_probs,
                lengths,
                self.labels,
                self.blank,
                beam_width=width,
                lm_scorer=lm_scorer,
                lm_weight=weight,
            )
        return _ctc_greedy_decode(log_probs, lengths, self.labels, self.blank)

    @torch.no_grad()
    def transcribe_batch(
        self,
        waveforms: Sequence[torch.Tensor],
        *,
        sample_rate: Optional[int] = None,
        use_beam_search: bool = False,
        beam_width: Optional[int] = None,
        lm_scorer: Optional[LanguageModelScorer] = None,
        lm_weight: Optional[float] = None,
        return_log_probs: bool = False,
    ) -> List[TranscriptionResult]:
        self._ensure_sample_rate(sample_rate)

        features = batch_compute_mel_spectrogram(waveforms, self.feature_config)
        ctc_input = prepare_ctc_inputs(features, device=self.device)
        logits, lengths = self.model(ctc_input.features, ctc_input.lengths)
        log_probs = F.log_softmax(logits, dim=-1)

        transcripts = self._decode(
            log_probs,
            lengths,
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            lm_scorer=lm_scorer,
            lm_weight=lm_weight,
        )

        results: List[TranscriptionResult] = []
        for idx, transcript in enumerate(transcripts):
            lp = log_probs[idx, : lengths[idx]] if return_log_probs else None
            results.append(TranscriptionResult(transcript=transcript, log_probs=lp))
        return results

    @torch.no_grad()
    def transcribe_stream(
        self,
        stream: Iterable[torch.Tensor],
        *,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[float] = None,
        use_beam_search: bool = False,
        beam_width: Optional[int] = None,
        lm_scorer: Optional[LanguageModelScorer] = None,
        lm_weight: Optional[float] = None,
    ) -> Iterable[TranscriptionResult]:
        """Yield incremental transcripts for a stream of waveform chunks."""

        self._ensure_sample_rate(sample_rate)
        buffer: List[torch.Tensor] = []
        target_samples = None
        if chunk_size is not None:
            target_samples = int(chunk_size * self.feature_config.sample_rate)
        elif self.inference_config.streaming_chunk_seconds > 0:
            target_samples = int(
                self.inference_config.streaming_chunk_seconds
                * self.feature_config.sample_rate
            )

        for chunk in stream:
            buffer.append(chunk)
            waveform = torch.cat(buffer, dim=-1)

            if target_samples is not None and waveform.size(-1) < target_samples:
                self.logger.debug(
                    "Accumulated %d/%d samples before decoding",
                    waveform.size(-1),
                    target_samples,
                )
                continue

            results = self.transcribe_batch(
                [waveform],
                sample_rate=sample_rate,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
                lm_scorer=lm_scorer,
                lm_weight=lm_weight,
                return_log_probs=False,
            )
            yield results[0]
            self.logger.debug(
                "Produced streaming transcript of length %d characters",
                len(results[0].transcript),
            )
            buffer = []

        if buffer:
            waveform = torch.cat(buffer, dim=-1)
            results = self.transcribe_batch(
                [waveform],
                sample_rate=sample_rate,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
                lm_scorer=lm_scorer,
                lm_weight=lm_weight,
                return_log_probs=False,
            )
            yield results[0]


__all__ = ["SpeechRecognizer", "TranscriptionResult", "LanguageModelScorer"]

