"""Wrappers and download helpers for public speech datasets.

The functions defined in this module use either :mod:`datasets` from the
🤗 HuggingFace ecosystem or :mod:`torchaudio` depending on what is
available in the execution environment. Each helper ensures that the
requested split is cached on disk so repeated runs do not re-download the
corpus.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

try:  # pragma: no cover - optional dependency
    import datasets as hf_datasets  # type: ignore
except Exception:  # pragma: no cover - keep the public API working
    hf_datasets = None

try:  # pragma: no cover - optional dependency
    from torchaudio.datasets import COMMONVOICE, LIBRISPEECH
except Exception:  # pragma: no cover - keep the public API working
    LIBRISPEECH = None
    COMMONVOICE = None

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetHandle:
    """Descriptor for a prepared dataset.

    Attributes
    ----------
    path:
        Directory where the dataset is cached.
    name:
        Human readable name of the dataset.
    subset:
        Optional subset identifier (e.g. ``"train.clean.100"`` for
        LibriSpeech).
    """

    path: Path
    name: str
    subset: Optional[str] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)


def _ensure_cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    cache_path = Path(cache_dir or Path.home() / ".cache" / "speech_datasets")
    cache_path.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Using dataset cache directory: %s", cache_path)
    return cache_path


def download_librispeech(
    split: str = "train.clean.100",
    cache_dir: Optional[Union[str, Path]] = None,
    use_huggingface: Optional[bool] = None,
) -> DatasetHandle:
    """Download the LibriSpeech dataset split.

    Parameters
    ----------
    split:
        Dataset split to download (e.g. ``"train.clean.100"``).
    cache_dir:
        Directory where the dataset should be cached.
    use_huggingface:
        Force using the HuggingFace datasets library. ``None`` lets the
        function decide based on availability.
    """

    cache_path = _ensure_cache_dir(cache_dir) / "librispeech"
    cache_path.mkdir(parents=True, exist_ok=True)

    if use_huggingface is None:
        use_huggingface = hf_datasets is not None

    if use_huggingface:
        if hf_datasets is None:  # pragma: no cover - safety net
            raise RuntimeError("datasets library is not installed")
        LOGGER.info("Downloading LibriSpeech split '%s' via datasets", split)
        hf_datasets.load_dataset(  # type: ignore[call-arg]
            "librispeech_asr", split=split, cache_dir=str(cache_path)
        )
    else:
        if LIBRISPEECH is None:  # pragma: no cover - safety net
            raise RuntimeError("torchaudio is not installed")
        LOGGER.info("Downloading LibriSpeech split '%s' via torchaudio", split)
        LIBRISPEECH(root=str(cache_path), url=split, download=True)

    return DatasetHandle(path=cache_path, name="LibriSpeech", subset=split)


def download_common_voice(
    language: str = "en",
    version: str = "7.0",
    cache_dir: Optional[Union[str, Path]] = None,
    use_huggingface: Optional[bool] = None,
) -> DatasetHandle:
    """Download the Mozilla Common Voice dataset for the given language."""

    cache_path = _ensure_cache_dir(cache_dir) / f"common_voice_{language}"
    cache_path.mkdir(parents=True, exist_ok=True)

    if use_huggingface is None:
        use_huggingface = hf_datasets is not None

    if use_huggingface:
        if hf_datasets is None:  # pragma: no cover - safety net
            raise RuntimeError("datasets library is not installed")
        LOGGER.info(
            "Downloading Common Voice (%s) version %s via datasets", language, version
        )
        hf_datasets.load_dataset(  # type: ignore[call-arg]
            "mozilla-foundation/common_voice_11_0",
            language,
            cache_dir=str(cache_path),
            revision=version,
        )
    else:
        if COMMONVOICE is None:  # pragma: no cover - safety net
            raise RuntimeError("torchaudio is not installed")
        LOGGER.info(
            "Downloading Common Voice (%s) version %s via torchaudio",
            language,
            version,
        )
        COMMONVOICE(root=str(cache_path), tsv=language, version=version, download=True)

    return DatasetHandle(path=cache_path, name="Common Voice", subset=language)


def ensure_dataset(
    name: str,
    *,
    split: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    language: str = "en",
    version: str = "7.0",
    use_huggingface: Optional[bool] = None,
) -> DatasetHandle:
    """Ensure that a supported dataset is available locally.

    Parameters
    ----------
    name:
        Supported dataset identifier (``"librispeech"`` or ``"common_voice"``).
    split, language, version:
        Parameters forwarded to the specific download helper.
    cache_dir:
        Location where downloaded archives should be stored.
    use_huggingface:
        Force using the HuggingFace ``datasets`` backend.
    """

    name = name.lower()
    if name == "librispeech":
        if split is None:
            split = "train.clean.100"
        return download_librispeech(
            split=split, cache_dir=cache_dir, use_huggingface=use_huggingface
        )
    if name in {"common_voice", "common-voice", "cv"}:
        return download_common_voice(
            language=language,
            version=version,
            cache_dir=cache_dir,
            use_huggingface=use_huggingface,
        )

    raise ValueError(f"Unsupported dataset '{name}'")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache speech datasets")
    parser.add_argument("--dataset", required=True, help="Dataset identifier (librispeech, common_voice)")
    parser.add_argument("--split", help="Dataset split (LibriSpeech only)")
    parser.add_argument("--language", default="en", help="Language code for Common Voice")
    parser.add_argument("--version", default="7.0", help="Dataset version for Common Voice")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Directory to cache downloads")
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Force using the HuggingFace datasets backend",
    )
    return parser.parse_args(argv)


def _main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    handle = ensure_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        language=args.language,
        version=args.version,
        use_huggingface=True if args.use_huggingface else None,
    )
    print(f"Dataset '{handle.name}' ready at {handle.path}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main()
