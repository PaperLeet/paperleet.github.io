"""Settings for the standalone local edition."""
import os
from pathlib import Path
from paperleet.catalog import root


def data_directory():
    return Path(os.environ.get("PAPERLEET_DATA_DIR", root() / ".paperleet")).expanduser().resolve()


def validate_local_settings():
    if os.environ.get("PAPERLEET_MODE", "local") != "local":
        raise ValueError("The open edition only runs locally. Use the separate online application for hosted services.")
