"""Business logic helpers for user voice training workflows."""
from pathlib import Path
from typing import Optional, Sequence

from audio_utils import merge_user_voices, user_profile_dir


def train_new_voice(
    user_id: int,
    *,
    selected_clips: Optional[Sequence[Path | str]] = None,
    max_references: int = 3,
) -> Optional[list[str]]:
    """Build a new voice profile from freshly recorded samples.

    Returns curated reference paths or ``None`` if nothing was selected.
    """

    profile_dir = user_profile_dir(user_id)
    references = merge_user_voices(
        user_id,
        profile_dir,
        selected_clips=selected_clips,
        max_references=max_references,
    )
    if not references:
        return None
    return [str(Path(ref)) for ref in references]


def continue_training(
    user_id: int,
    *,
    selected_clips: Optional[Sequence[Path | str]] = None,
    max_references: int = 3,
) -> Optional[list[str]]:
    """Update an existing profile by merging all saved samples.

    Returns curated reference paths or ``None`` if nothing was selected.
    """

    profile_dir = user_profile_dir(user_id)
    references = merge_user_voices(
        user_id,
        profile_dir,
        selected_clips=selected_clips,
        max_references=max_references,
    )
    if not references:
        return None
    return [str(Path(ref)) for ref in references]
