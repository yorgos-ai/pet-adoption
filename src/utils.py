from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


def get_artifact_path() -> Path:
    """Returns the path to the artifact folder."""
    project_root = get_project_root()
    return project_root / "artifacts"
