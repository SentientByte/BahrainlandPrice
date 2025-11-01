import os
from pathlib import Path


def get_project_paths():
    base_dir = Path(__file__).resolve().parent
    venv_dir = base_dir / ".venv"
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    return base_dir, venv_dir, data_dir, output_dir


def ensure_dirs():
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
