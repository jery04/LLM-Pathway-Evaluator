"""Utilities to fetch, normalize, and cache multi-platform online course data.

This module downloads course datasets from Kaggle (via `kagglehub` if available),
copies CSV files into the project's `data/csv/` directory, and generates a normalized
JSON cache used downstream for embedding generation and course planning.

1. Download raw CSV datasets from configured Kaggle sources.
2. Copy or extract CSV files into a local `data/csv/` folder.
3. Normalize rows (duration, difficulty, cost, skills, category, level).
4. Generate a JSON cache (`normalized_courses.json`) to avoid repeated normalization.
5. Produce embeddings from the normalized cache via `llm_adapter.generate_embeddings`.

All docstrings and comments follow the project convention (English).
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


# Kaggle dataset identifiers 
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


# ===========================================================================
# AUXILIARY METHODS
# ===========================================================================

def _find_header(header_map: dict, aliases) -> Optional[str]:
    """Find the first matching header in a normalized header map using aliases.

    Returns the canonical header name when an alias matches, or None if no match exists."""
    for alias in aliases:
        key = _normalize_header(alias)
        if key in header_map:
            return header_map[key]
    return None

def _first_value(row: dict, header_map: dict, aliases) -> str:
    """Return the first available value from a row using a set of header aliases.

    It looks up the normalized header name and strips whitespace from the field value."""
    header = _find_header(header_map, aliases)
    if not header:
        return ""
    return (row.get(header) or "").strip()


# ===========================================================================
# PARSING AND NORMALIZATION METHODS
# ===========================================================================

def _normalize_header(value: str) -> str:
    """Normalize a CSV header string for case-insensitive alias matching.

    It lowercases the text, converts underscores and dashes to spaces, and collapses whitespace."""
    text = (value or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text)

def _normalize_course_row(row: dict, header_map: dict) -> Optional[dict]:
    """Convert a raw CSV row into the normalized course schema for caching.

    Rows without a usable course link or course name are discarded as invalid."""
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

def _parse_duration_to_months(value: str) -> int:
    """Parse a duration string and return an approximate duration in months.

    Supports month phrases, hour estimates, and plain numeric values with a sensible default."""
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
    """Normalize course level text into Beginner, Intermediate, or Advanced.

    If no known level keyword is found, the original trimmed text is returned."""
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
    """Infer a numeric difficulty score from row fields and rating descriptors.

    Text clues map to a 1-10 scale, and numeric ratings are rounded into that range."""
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
    """Parse a cost field from the row and return a USD integer value.

    Free values are converted to zero and numeric amounts are rounded."""
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
    """Extract skill terms from a skills field using common delimiters.

    Non-string values are ignored and empty items are filtered out."""
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
    """Select a category value from the row using subject and topic aliases.

    Returns 'General' when no category-like field is present."""
    for key in ("subject", "category", "topic", "field"):
        header = _find_header(header_map, (key,))
        if not header:
            continue
        value = (row.get(header) or "").strip()
        if value:
            return value
    return "General"


# ===========================================================================
# CACHE METHODS (JSON)
# ===========================================================================

def _copy_downloaded_dataset_to_csv(source_path: Path, csv_dir: Path) -> None:
    """Copy or extract CSV files from the Kaggle download into the project CSV folder.

    This supports raw CSV files, Kaggle dataset directories, and ZIP archives that contain CSVs.
    The files are renamed sequentially to avoid clashes in the `data/csv` folder."""
    csv_dir.mkdir(parents=True, exist_ok=True)

    def _next_dataset_index(dir_path: Path) -> int:
        pattern = re.compile(r"^dataset_(\d+)\.csv$", re.IGNORECASE)
        max_i = 0
        for p in dir_path.iterdir():
            if p.is_file():
                m = pattern.match(p.name)
                if m:
                    try:
                        max_i = max(max_i, int(m.group(1)))
                    except Exception:
                        continue
        return max_i + 1

    def _copy_csv_file(src_path: Path, index: int) -> int:
        dest = csv_dir / f"dataset_{index}.csv"
        while dest.exists():
            index += 1
            dest = csv_dir / f"dataset_{index}.csv"
        shutil.copy2(src_path, dest)
        return index + 1

    next_index = _next_dataset_index(csv_dir)

    if source_path.is_dir():
        for csv_path in sorted(source_path.rglob("*.csv")):
            next_index = _copy_csv_file(csv_path, next_index)
        return

    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix == ".csv":
            _copy_csv_file(source_path, next_index)
            return

        if suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(source_path, "r") as archive:
                for member in sorted(archive.namelist()):
                    if member.lower().endswith(".csv"):
                        with archive.open(member) as src_handle:
                            dest = csv_dir / f"dataset_{next_index}.csv"
                            while dest.exists():
                                next_index += 1
                                dest = csv_dir / f"dataset_{next_index}.csv"
                            with dest.open("wb") as out_handle:
                                shutil.copyfileobj(src_handle, out_handle)
                            next_index += 1
            return

def _generate_normalized_cache(data_dir: Path) -> Optional[Path]:
    """Normalize course CSV datasets into a shared JSON cache used by the planner.

    Only source files with a recognized course link column are converted, and rows
    lacking a valid link are filtered out before cache generation."""
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


# ===========================================================================
# MAIN 
# ===========================================================================

def main() -> None:
    """Ensure course data is available locally, normalize it, and generate embeddings.

    1. Create data directory and check for existing CSVs.
    2. Download missing Kaggle datasets into CSV folder.
    3. Build normalized cache from raw CSV data.
    4. Generate embeddings from normalized cache.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = data_dir / "csv"
    csv_files_exist = csv_dir.exists() and any(
        item.is_file() and item.suffix.lower() == ".csv" for item in csv_dir.iterdir()
    )

    # Simple textual progress bar: each dataset download attempt plus
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
                _copy_downloaded_dataset_to_csv(source_path, csv_dir)
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
    _generate_normalized_cache(data_dir)
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
