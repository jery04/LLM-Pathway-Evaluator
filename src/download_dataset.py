"""Utilities to fetch and prepare the course dataset for the project.

This module attempts to download a multi-platform online courses dataset
from Kaggle (via `kagglehub` if available), copy files into the project's
`data/` directory, and generate a normalized JSON cache used by the
`planner` module to avoid repeated normalization work.

All docstrings and comments are written in English per project convention.
"""

from pathlib import Path  # Path class for filesystem path manipulation
import shutil  # high-level file operations (copytree, copy2)
from typing import Optional # Optional type hint for functions that may return None

try:
    import kagglesdk.kaggle_env as kaggle_env # Import Kaggle environment utilities to check if running in Kaggle notebook

    # If the Kaggle SDK is missing the `get_web_endpoint` function, but has `get_endpoint`, we alias it to maintain compatibility.
    if not hasattr(kaggle_env, "get_web_endpoint") and hasattr(kaggle_env, "get_endpoint"):
        kaggle_env.get_web_endpoint = kaggle_env.get_endpoint

    import kagglehub # Import KaggleHub for dataset download if available
except Exception:
    # If either import fails, set `kagglehub` to None and proceed using local data
    kagglehub = None


# Kaggle dataset identifier to download if `kagglehub` is available.
KAGGLE_DATASET_ID = "everydaycodings/multi-platform-online-courses-dataset"


def copy_dataset_to_data(source_dir: Path, target_dir: Path) -> None:
    """Copy dataset files from source_dir into target_dir.
    Merges directories and overwrites files when needed.
    """
    # Ensure the target directory exists before copying
    target_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each top-level item in the downloaded dataset folder
    for item in source_dir.iterdir():
        destination = target_dir / item.name

        # Use copytree for directories (preserve subfolders), copy2 for files
        if item.is_dir():
            # `dirs_exist_ok=True` merges directories without raising an error
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            # copy2 preserves metadata like modification times
            shutil.copy2(item, destination)

def data_has_csv_files(data_dir: Path) -> bool:
    """Return True if data_dir contains any top-level CSV file.
    Used as a quick check before trying a Kaggle download.
    """
    return data_dir.exists() and any(
        item.is_file() and item.suffix.lower() == ".csv" for item in data_dir.iterdir()
    )

def _generate_normalized_cache(project_root: Path, data_dir: Path) -> Optional[Path]:
    """Build normalized_courses.json from CSV files using planner helpers.
    Return the cache path on success, or None on failure.
    """
    try:
        import importlib.util
        import json

        # Locate the planner module in `src/planner.py` and import it by path
        planner_path = project_root / "src" / "planner.py"
        spec = importlib.util.spec_from_file_location("local_planner", str(planner_path))
        planner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(planner)

        # Use planner's internal helpers to load and normalize records
        records = planner._load_records_from_path(data_dir)
        normalized = planner._finalize_records(records)

        cache = data_dir / "normalized_courses.json"
        # If the normalized cache already exists, do not regenerate it.
        if cache.exists():
            # Return the existing cache path to signal no action taken
            return cache.resolve()

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cache.resolve()
    except Exception:
        # Fail silently here; caller will decide how to surface the error
        return None

def main() -> None:
    """Main entry point to ensure dataset is available and cached.

    Steps performed:
    1. Ensure the project's `data/` directory exists.
    2. If no CSVs are present and `kagglehub` is available, attempt download.
    3. Copy downloaded files into `data/` (if download succeeded).
    4. Attempt to build a normalized JSON cache using `planner.py`.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # If there are no CSV files in data/, and we have kagglehub, try fetching
    if not data_has_csv_files(data_dir) and kagglehub:
        try:
            # `kagglehub.dataset_download` should return a filesystem path
            path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
            source_path = Path(path)
            print(f"· Path to dataset files: {source_path}")

            # Copy downloaded files into the project's data directory
            copy_dataset_to_data(source_path, data_dir)
            print(f"· Dataset copied to: {data_dir}")
        except Exception as e:
            # If download or copy fails, log and continue using any existing data
            print("· kagglehub download failed, will use existing data/:", e)
    elif data_has_csv_files(data_dir):
        # CSV files already present — no need to download anything
        print(f"· CSV files already present in {data_dir}; skipping Kaggle download")
    else:
        # No kagglehub and no CSVs — proceed and let later steps fail gracefully
        print("· kagglehub not available; using existing files under data/ if present")

    # After copying raw data, attempt to generate a normalized cache file.
    cache_path = _generate_normalized_cache(project_root, data_dir)
    if cache_path:
        print(f"· Normalized cache written to: {cache_path}")
    else:
        print("· Could not write normalized cache: planner import or normalization failed")


if __name__ == "__main__":
    main()
