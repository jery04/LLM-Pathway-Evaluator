"""Utilities to fetch and prepare the course dataset for the project.

This module attempts to download a multi-platform online courses dataset
from Kaggle (via `kagglehub` if available), copy files into the project's
`data/` directory, and generate a normalized JSON cache used by the
`planner` module to avoid repeated normalization work.

All docstrings and comments are written in English per project convention.
"""

from pathlib import Path  # Path class for filesystem path manipulation
import shutil             # high-level file operations (copytree, copy2)
from typing import Optional  # Optional type hint helper
import csv                # CSV parsing for source datasets
import re                 # Provides regular expressions for pattern matching
import json               # Handles JSON serialization and parsing
from llm_adapter import generate_embeddings # Functions for embeddings and prerequisite inference

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

# Column header aliases for link and name fields.
LINK_COLUMN_ALIASES = (
    "link",
    "url",
    "course url",
    "course_url",
    "courseurl",
)

NAME_COLUMN_ALIASES = (
    "name",
    "title",
    "course",
    "course title",
    "course name",
)


#--------------------------------------------------------------
# METHODS 
#--------------------------------------------------------------

def _normalize_header(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text)

def _find_header(header_map: dict, aliases) -> Optional[str]:
    for alias in aliases:
        key = _normalize_header(alias)
        if key in header_map:
            return header_map[key]
    return None

def _first_value(row: dict, header_map: dict, aliases) -> str:
    header = _find_header(header_map, aliases)
    if not header:
        return ""
    return (row.get(header) or "").strip()

def _parse_duration_to_months(value: str) -> int:
    text = (value or "").strip().lower()
    if not text:
        return 1

    month_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:[-to]+\s*(\d+(?:[.,]\d+)?)\s*)?month", text)
    if month_match:
        first = float(month_match.group(1).replace(",", "."))
        second = month_match.group(2)
        if second:
            second_value = float(second.replace(",", "."))
            return max(1, int(round((first + second_value) / 2)))
        return max(1, int(round(first)))

    hour_match = re.search(r"(\d+(?:[.,]\d+)?)\s*hour", text)
    if hour_match:
        hours = float(hour_match.group(1).replace(",", "."))
        return max(1, int(round(hours / 40.0)))

    number_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if number_match:
        return max(1, int(round(float(number_match.group(1).replace(",", ".")))))

    return 1

def _parse_level(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    if "begin" in text or "intro" in text:
        return "Beginner"
    if "inter" in text or "medium" in text:
        return "Intermediate"
    if "adv" in text or "expert" in text:
        return "Advanced"
    return value.strip()

def _parse_difficulty(row: dict, header_map: dict) -> int:
    for key in ("difficulty level", "difficulty", "level", "rating", "course rating"):
        header = _find_header(header_map, (key,))
        if not header:
            continue
        raw = (row.get(header) or "").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if any(token in lowered for token in ("begin", "intro")):
            return 3
        if any(token in lowered for token in ("inter", "mid")):
            return 6
        if any(token in lowered for token in ("adv", "expert")):
            return 8
        numeric = re.search(r"(\d+(?:[.,]\d+)?)", raw)
        if numeric:
            value = float(numeric.group(1).replace(",", "."))
            return max(1, min(10, int(round(value))))
    return 5

def _parse_cost(row: dict, header_map: dict) -> int:
    for key in ("price", "cost", "cost usd", "price usd"):
        header = _find_header(header_map, (key,))
        if not header:
            continue
        raw = (row.get(header) or "").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"free", "0", "$0", "0.0"}:
            return 0
        numeric = re.search(r"(\d+(?:[.,]\d+)?)", raw)
        if numeric:
            return int(round(float(numeric.group(1).replace(",", "."))))
    return 0

def _parse_skills(row: dict, header_map: dict) -> list[str]:
    for key in ("skills", "associatedskills", "skill", "course skills"):
        header = _find_header(header_map, (key,))
        if not header:
            continue
        raw = row.get(header) or ""
        if not isinstance(raw, str):
            continue
        cleaned = raw.replace("{", "").replace("}", "").replace('"', "")
        pieces = re.split(r"[,;|/]", cleaned)
        skills = [piece.strip() for piece in pieces if piece and piece.strip()]
        if skills:
            return skills
    return []

def _parse_category(row: dict, header_map: dict) -> str:
    for key in ("subject", "category", "topic", "field"):
        header = _find_header(header_map, (key,))
        if not header:
            continue
        value = (row.get(header) or "").strip()
        if value:
            return value
    return "General"

def _normalize_course_row(row: dict, header_map: dict) -> Optional[dict]:
    link = _first_value(row, header_map, LINK_COLUMN_ALIASES)
    if not link:
        return None

    name = _first_value(row, header_map, NAME_COLUMN_ALIASES)
    if not name:
        return None

    duration_raw = _first_value(
        row,
        header_map,
        ("duration", "course duration", "content_duration", "duration months", "duration_months"),
    )
    level_raw = _first_value(row, header_map, ("level", "difficulty level"))

    return {
        "name": name,
        "duration_months": _parse_duration_to_months(duration_raw),
        "difficulty": _parse_difficulty(row, header_map),
        "category": _parse_category(row, header_map),
        "cost_usd": _parse_cost(row, header_map),
        "level": _parse_level(level_raw),
        "skills": _parse_skills(row, header_map),
        "link": link,
    }

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

def _generate_normalized_cache(project_root: Path, data_dir: Path) -> Optional[Path]:
    """Build normalized_courses.json only from CSVs that expose a link column.

    Only datasets whose header includes a link/url/course URL column are used.
    Any row without a usable link is also discarded.
    """
    try:
        transformed = []
        csv_dir = data_dir / "csv"
        for csv_path in sorted(csv_dir.glob("dataset_*.csv")) if csv_dir.exists() else []:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header_map = {_normalize_header(name): name for name in (reader.fieldnames or []) if name}
                if not _find_header(header_map, LINK_COLUMN_ALIASES):
                    continue

                for row in reader:
                    normalized = _normalize_course_row(row, header_map)
                    if normalized:
                        transformed.append(normalized)

        cache = data_dir / "normalized_courses.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cache.resolve()
    except Exception:
        # Fail silently here; caller will decide how to surface the error
        return None


#--------------------------------------------------------------
# MAIN 
#--------------------------------------------------------------
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

    csv_dir = data_dir / "csv"
    csv_files_exist = csv_dir.exists() and any(
        item.is_file() and item.suffix.lower() == ".csv" for item in csv_dir.iterdir()
    )

    # Simple textual progress bar: each dataset download attempt plus
    # the normalization step is a unit of work. Compute total steps
    # based on whether `kagglehub` is available. This stays constant
    # so percentages don't exceed 100% when skipping work.
    # Compute total steps dynamically so progress reflects actual datasets.
    csv_dir = data_dir / "csv"
    csv_count = 0
    if csv_dir.exists():
        csv_count = sum(1 for _ in csv_dir.glob("dataset_*.csv"))

    if kagglehub:
        # Count download attempts as max between configured datasets and existing CSVs,
        # plus one normalization step.
        total_steps = max(len(KAGGLE_DATASETS), csv_count)
    else:
        # No downloads; at least the normalization step counts as one.
        total_steps = max(1, csv_count)

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
        _print_progress(100, f"processed {completed}/{total_steps}")
    elif kagglehub:
        for dataset_id in KAGGLE_DATASETS:
            try:
                source_path = Path(kagglehub.dataset_download(dataset_id))
                copy_dataset_to_data(source_path, data_dir)
            except Exception:
                pass
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
    # Only increment and print progress if we haven't already marked all steps completed
    if completed < total_steps:
        completed += 1
        percent = int(completed / total_steps * 100)
        _print_progress(percent, f"processed {completed}/{total_steps}")

    # Generate embeddings only when the cache is missing.
    print("\n\nGenerating embeddings for courses...")
    generate_embeddings(data_dir)

    # Finalize
    _final_success_message()

if __name__ == "__main__":
    main()
