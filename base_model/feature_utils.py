"""Utility definitions shared across feature-selection and data-loading scripts."""
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Acoustic feature families supported by the pipeline. Update this list when a
# new family is added to `load_ravdess_features`.
AVAILABLE_FAMILIES: List[str] = [
    "mfcc",       # 13-dimensional Mel-cepstral coefficients
    "spectral",   # Centroid, roll-off, bandwidth (3-D)
    "prosodic",   # Zero-crossing rate + RMS energy (2-D)
    "chroma",     # 12-bin chromagram
    "combined",   # Concatenation of all above
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_filenames(data_dir: str) -> List[str]:
    """Return absolute paths of all RAVDESS *.wav files in a deterministic order.

    Actor folders (Actor_XX) are traversed lexically; files within each actor
    are also sorted.  The returned list aligns with the order produced by
    `load_ravdess_features`, ensuring feature matrices, labels, and filenames
    stay synchronised.
    """
    root = Path(data_dir)
    actor_dirs = sorted(root.glob("Actor_*"))
    wav_paths = []

    if actor_dirs:
        for actor in actor_dirs:
            wav_paths.extend(sorted(actor.glob("*.wav")))
    else:
        wav_paths = sorted(root.glob("*.wav"))

    # Convert Path objects to strings for JSON-serialisation convenience
    return [str(p) for p in wav_paths]
