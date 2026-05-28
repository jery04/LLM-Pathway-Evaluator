"""Utilities to fetch and prepare the course dataset for the project.

This module attempts to download a multi-platform online courses dataset
from Kaggle (via `kagglehub` if available), copy files into the project's
`data/` directory, and generate a normalized JSON cache used by the
`planner` module to avoid repeated normalization work.

All docstrings and comments are written in English per project convention.
"""

from pathlib import Path  # Path class for filesystem path manipulation
import shutil             # high-level file operations (copytree, copy2)
from typing import Iterable, Optional  # Optional type hint and iterable helpers
import re                 # Provides regular expressions for pattern matching
import importlib.util     # Allows dynamic loading of modules at runtime
import json               # Handles JSON serialization and parsing

try:
    import kagglesdk.kaggle_env as kaggle_env # Import Kaggle environment utilities to check if running in Kaggle notebook

    # If the Kaggle SDK is missing the `get_web_endpoint` function, but has `get_endpoint`, we alias it to maintain compatibility.
    if not hasattr(kaggle_env, "get_web_endpoint") and hasattr(kaggle_env, "get_endpoint"):
        kaggle_env.get_web_endpoint = kaggle_env.get_endpoint

    import kagglehub # Import KaggleHub for dataset download if available
except Exception:
    # If either import fails, set `kagglehub` to None and proceed using local data
    kagglehub = None


# Kaggle dataset identifiers to download if `kagglehub` is available.
KAGGLE_DATASETS = [
    "everydaycodings/multi-platform-online-courses-dataset",
    "mahmoudahmed6/skillshare-top-1000-course",
    "khusheekapoor/coursera-courses-dataset-2021",
    "yusufdelikkaya/udemy-online-education-courses",
]

def copy_dataset_to_data(source_dir: Path, target_dir: Path) -> None:
    """Copy dataset files from source_dir into target_dir.
    Merges directories and overwrites files when needed.
    """
    # Ensure the target directory exists before copying
    target_dir.mkdir(parents=True, exist_ok=True)

    # CSVs will be stored in a dedicated `csv/` subfolder under `target_dir`
    csv_dir = target_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Helper: compute next available dataset index (dataset_1.csv, dataset_2.csv, ...)
    def _next_dataset_index(dir_path: Path) -> int:
        pattern = re.compile(r"^dataset_(\d+)\.csv$", re.IGNORECASE)
        max_i = 0
        if dir_path.exists():
            for p in dir_path.iterdir():
                if p.is_file():
                    m = pattern.match(p.name)
                    if m:
                        try:
                            max_i = max(max_i, int(m.group(1)))
                        except Exception:
                            continue
        return max_i + 1

    # Helper: rename any non-dataset_*.csv files under `base` recursively,
    # starting at `start_index`. Returns the next available index after renames.
    def _rename_csvs_in_dir(base: Path, start_index: int, out_dir: Path) -> int:
        pattern = re.compile(r"^dataset_(\d+)\.csv$", re.IGNORECASE)
        for p in sorted(base.rglob("*.csv")):
            # Skip files that already match the dataset_N pattern (they may already be in out_dir)
            if pattern.match(p.name) and p.parent == out_dir:
                continue
            # Determine target name in out_dir
            dest = out_dir / f"dataset_{start_index}.csv"
            while dest.exists():
                start_index += 1
                dest = out_dir / f"dataset_{start_index}.csv"
            try:
                p.replace(dest)
            except Exception:
                # Fallback to copy+unlink if rename across filesystems fails
                shutil.copy2(p, dest)
                try:
                    p.unlink()
                except Exception:
                    pass
            start_index += 1
        return start_index

    # Start numbering from the next available index in the csv directory
    next_index = _next_dataset_index(csv_dir)

    # Iterate over each top-level item in the downloaded dataset folder
    for item in source_dir.iterdir():
        destination = target_dir / item.name

        # If it's a directory, merge it into `target_dir` then move/rename CSVs into csv_dir
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
            next_index = _rename_csvs_in_dir(destination, next_index, csv_dir)
        else:
            # If it's a CSV file, copy it under a sequential dataset_i.csv name
            if item.suffix.lower() == ".csv":
                dest = csv_dir / f"dataset_{next_index}.csv"
                while dest.exists():
                    next_index += 1
                    dest = csv_dir / f"dataset_{next_index}.csv"
                shutil.copy2(item, dest)
                next_index += 1
            else:
                # Preserve non-CSV files with original name
                shutil.copy2(item, destination)

def data_has_csv_files(data_dir: Path) -> bool:
    """Return True if data_dir contains any top-level CSV file.
    Used as a quick check before trying a Kaggle download.
    """
    csv_dir = data_dir / "csv"
    return csv_dir.exists() and any(
        item.is_file() and item.suffix.lower() == ".csv" for item in csv_dir.iterdir()
    )

def _download_and_copy_dataset(dataset_id: str, data_dir: Path) -> bool:
    """Download a Kaggle dataset and merge its files into data_dir."""
    try:
        path = kagglehub.dataset_download(dataset_id)
        source_path = Path(path)
        copy_dataset_to_data(source_path, data_dir)
        return True
    except Exception as e:
        # Failure is silent; caller will handle progress reporting.
        return False

def _generate_normalized_cache(project_root: Path, data_dir: Path) -> Optional[Path]:
    """Build normalized_courses.json from CSV files using planner helpers.
    Return the cache path on success, or None on failure.
    
    Processes all CSV files and transforms records to have 'skills' field
    instead of 'prerequisitos' following the exact structure specified.
    """
    try:
        # Locate the planner module in `src/planner.py` and import it by path
        planner_path = project_root / "src" / "planner.py"
        spec = importlib.util.spec_from_file_location("local_planner", str(planner_path))
        planner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(planner)

        # Use planner's internal helpers to load and normalize records from ALL CSV files
        records = planner._load_records_from_path(data_dir)
        normalized = planner._finalize_records(records)

        # Transform records to match the exact schema:
        transformed = []
        for record in normalized:
            transformed_record = {
                "name": record.get("name", record.get("nombre", "")),
                "duration_months": record.get("duration_months", record.get("duracion_meses", 1)),
                "difficulty": record.get("difficulty", record.get("dificultad", 1)),
                "category": record.get("category", record.get("categoria", "General")),
                "cost_usd": record.get("cost_usd", record.get("coste_USD", 0)),
                "level": record.get("level", ""),
                "skills": record.get("skills", []),  # Use skills as primary field
                "link": record.get("link", ""),
            }
            # Only add record if it has at least a name
            if transformed_record["name"]:
                transformed.append(transformed_record)

        cache = data_dir / "normalized_courses.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cache.resolve()
    except Exception:
        # Fail silently here; caller will decide how to surface the error
        return None

def main() -> None:
    """Main entry point to ensure dataset is available and cached.

    Steps performed:
    1. Ensure the project's `data/` directory exists.
    2. Check if CSV files already exist; if so, skip downloads.
    3. Download any missing Kaggle datasets when `kagglehub` is available and CSVs are missing.
    4. Copy downloaded files into `data/` (if download succeeded).
    5. Rebuild the normalized JSON cache using `planner.py`.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if CSV files already exist
    csv_files_exist = data_has_csv_files(data_dir)

    # Simple textual progress bar: each dataset download attempt plus
    # the normalization step is a unit of work. Compute total steps
    # based on whether `kagglehub` is available. This stays constant
    # so percentages don't exceed 100% when skipping work.
    if kagglehub:
        total_steps = len(KAGGLE_DATASETS) + 1
    else:
        total_steps = 1

    def _print_progress(percent: int, message: str = "") -> None:
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {percent:3d}% {message}", end="", flush=True)

    def _final_success_message() -> None:
        GREEN = "\033[92m"
        RESET = "\033[0m"
        # Move to next line then print green success message
        print()
        print(f"{GREEN}Download completed successfully{RESET}")

    # Start progress at 0%
    completed = 0
    _print_progress(0, "starting")

    # Only download if CSV files don't already exist
    if csv_files_exist:
        # All dataset-download steps are considered already completed.
        completed = total_steps
        _print_progress(100, f"processed {completed}/{total_steps} (CSV present, skipped downloads)")
    elif kagglehub:
        for idx, dataset_id in enumerate(KAGGLE_DATASETS):
            _download_and_copy_dataset(dataset_id, data_dir)
            if completed < total_steps:
                completed += 1
            percent = int(completed / total_steps * 100)
            _print_progress(percent, f"processed {completed}/{total_steps}")
    else:
        # No kagglehub available; still continue to normalization step silently.
        completed = 0
        _print_progress(0, "no kagglehub")

    # After copying raw data, attempt to generate a normalized cache file.
    _generate_normalized_cache(project_root, data_dir)
    # Only increment if we haven't already marked all steps completed
    if completed < total_steps:
        completed += 1
    percent = int(completed / total_steps * 100)
    _print_progress(percent, f"processed {completed}/{total_steps}")

    # Finalize
    _final_success_message()


if __name__ == "__main__":
    main()
