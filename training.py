"""Business logic helpers for user voice training workflows."""
from pathlib import Path
from typing import Optional

from audio_utils import merge_user_voices, user_profile_path


def train_new_voice(user_id: int) -> Optional[str]:
    """Build a new voice profile from freshly recorded samples.

    Returns the path to the generated profile or ``None`` if nothing was merged.
    """

    profile_path = user_profile_path(user_id)
    merged = merge_user_voices(user_id, profile_path)
    if not merged:
        return None
    return str(Path(merged))


def continue_training(user_id: int) -> Optional[str]:
    """Update an existing profile by merging all saved samples.

    Returns the path to the generated profile or ``None`` if nothing was merged.
    """

    profile_path = user_profile_path(user_id)
    merged = merge_user_voices(user_id, profile_path)
    if not merged:
        return None
    return str(Path(merged))
