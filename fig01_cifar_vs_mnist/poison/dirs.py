__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "LOG_DIR",
    "MODELS_DIR",
    "PLOTS_DIR",
    "POISON_DIR",
    "RES_DIR",
]

from pathlib import Path


BASE_DIR = None
DATA_DIR = None
LOG_DIR = None
MODELS_DIR = None
PLOTS_DIR = None
POISON_DIR = None
RES_DIR = None


def _update_all_paths():
    r""" Sets all path names based on the base directory """
    global BASE_DIR, DATA_DIR, LOG_DIR, MODELS_DIR, PLOTS_DIR, POISON_DIR, RES_DIR

    BASE_DIR = Path(".").absolute()

    DATA_DIR = BASE_DIR / ".data"
    LOG_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    PLOTS_DIR = BASE_DIR / "plots"
    POISON_DIR = BASE_DIR / ".poison"
    RES_DIR = BASE_DIR / "res"


_update_all_paths()
