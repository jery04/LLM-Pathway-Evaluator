"""planner.py
Utilities for loading, normalizing and generating learning pathway
plans from a catalog of online courses. Functions include loaders,
normalizers, prerequisite inference, indexing and search heuristics.
"""

import csv                         # Reads and writes CSV (comma‑separated values) files
import heapq                       # Implements priority queues and heap‑based algorithms
import json                        # Loads and dumps JSON data structures
import re                          # Provides regular expressions for pattern matching
from pathlib import Path           # Object‑oriented filesystem path handling
from typing import List, Dict, Set, Tuple, Optional   # Type hints for common container types

# SECCION I/O and Loader ------------------------------------------------------
KAGGLE_DATASET_ID = "everydaycodings/multi-platform-online-courses-dataset"
KAGGLE_DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "kagglehub"

ROLE_KEYWORDS = {
    'AI Engineer': ['machine learning', 'deep learning', 'artificial intelligence', 'generative ai', 'python', 'statistics'],
    'Data Scientist': ['data science', 'machine learning', 'statistics', 'python', 'sql', 'data analysis'],
    'Cloud Engineer': ['cloud', 'devops', 'docker', 'kubernetes', 'linux', 'aws'],
    'Backend Engineer': ['backend', 'api', 'database', 'sql', 'python', 'javascript', 'node'],
    'Cybersecurity Analyst': ['cybersecurity', 'security', 'network', 'linux', 'cryptography'],
}

LEVEL_RANKS = {
    'beginner': 1,
    'introductory': 1,
    'fundamental': 1,
    'fundamentals': 1,
    'basic': 1,
    'intermediate': 2,
    'upper intermediate': 2,
    'advanced': 3,
    'expert': 4,
    'all levels': 2,
}

CATEGORY_KEYWORDS = [
    ('IA', ['machine learning', 'deep learning', 'artificial intelligence', 'generative ai', 'nlp']),
    ('Datos', ['data science', 'data analysis', 'analytics', 'sql', 'statistics', 'excel']),
    ('Cloud', ['cloud', 'aws', 'gcp', 'azure', 'devops', 'docker', 'kubernetes', 'linux']),
    ('Web', ['web', 'javascript', 'react', 'frontend', 'backend', 'html', 'css', 'node']),
    ('Matemáticas', ['math', 'mathematics', 'statistics', 'linear algebra', 'calculus', 'probability']),
    ('Seguridad', ['cybersecurity', 'security', 'cryptography', 'network']),
    ('Diseño', ['design', 'ux', 'ui', 'graphic', 'illustration', 'procreate', 'photoshop']),
    ('Management', ['project management', 'product management', 'leadership', 'strategy', 'planning']),
]


def load_courses(path: Optional[str] = None) -> List[Dict]:
    """Load normalized courses, using cache when available.

    If a cached normalized JSON exists it is returned; otherwise this
    function loads raw records and normalizes them before caching.
    """
    cache_file = Path(__file__).resolve().parent.parent / "data" / "normalized_courses.json"

    # If a cached normalized JSON exists, load and return it (single normalization run).
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and any(_extract_skills(item.get('skills')) for item in cached if isinstance(item, dict)):
                return cached
            # A cache with no skills is usually stale, so fall through and rebuild it.
        except Exception:
            # If cache is corrupted, fall through to re-generate it.
            pass

    # Otherwise, load raw records from the requested path, local data, or Kaggle,
    # normalize them once and persist the normalized JSON for future runs.
    if path:
        candidate = Path(path)
        if candidate.exists():
            records = _load_records_from_path(candidate)
            final = _finalize_records(records)
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            return final

    # First try to load any local data shipped with the repository (data/*.csv, json, jsonl).
    local_data_dir = Path(__file__).resolve().parent.parent / "data"
    if local_data_dir.exists():
        local_records = _load_records_from_path(local_data_dir)
        if local_records:
            final = _finalize_records(local_records)
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            return final

    # Fall back to attempting to download via kagglehub if local data isn't available.
    downloaded = _load_kaggle_dataset(KAGGLE_DATASET_ID)
    final = _finalize_records(downloaded)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return final

def _get_duration(course: Dict) -> int:
    """Return the course duration (months) from possible fields."""
    return int(course.get('duration_months') or course.get('duracion_meses') or course.get('duration') or 0)

def _get_cost(course: Dict) -> int:
    """Return the numeric cost for a course from common keys."""
    return int(course.get('cost_usd') or course.get('coste_USD') or course.get('cost') or 0)

def _get_difficulty(course: Dict) -> int:
    """Return stored difficulty value, supporting Spanish key names."""
    return int(course.get('difficulty') or course.get('dificultad') or 0)

def _get_category(course: Dict) -> str:
    """Return the category string for a course, defaulting to 'General'."""
    return course.get('category') or course.get('categoria') or 'General'

def _load_kaggle_dataset(dataset_id: str) -> List[Dict]:
    """Attempt to download a Kaggle dataset and load its records.

    Gracefully returns an empty list when the kaggle client is absent.
    """
    try:
        import kagglehub
    except ImportError:
        return []

    try:
        dataset_path = Path(kagglehub.dataset_download(dataset_id, output_dir=str(KAGGLE_DOWNLOAD_DIR)))
    except Exception:
        return []

    return _load_records_from_path(dataset_path)

def _load_records_from_path(path: Path) -> List[Dict]:
    """Load all records found under a path (file or directory).

    Recursively finds supported file types and returns a list of
    normalized records by delegating to `_load_records_from_file`.
    """
    if path.is_file():
        return _load_records_from_file(path, source_name=path.stem)

    records: List[Dict] = []
    for candidate in sorted(path.rglob('*')):
        # skip unsupported file types early
        if candidate.suffix.lower() not in {'.csv', '.json', '.jsonl'}:
            continue
        records.extend(_load_records_from_file(candidate, source_name=candidate.stem))
    return records

def _load_records_from_file(path: Path, source_name: str = '') -> List[Dict]:
    """Read a single file and return a list of normalized records.

    Supports CSV, JSON and JSONL formats. Each raw entry is passed
    through `_normalize_record` to produce a canonical record form.
    """
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
            return [_normalize_record(dict(row), source_name=source_name) for row in csv.DictReader(handle)]

    with open(path, 'r', encoding='utf-8') as handle:
        raw = handle.read().strip()
        if not raw:
            return []
        if suffix == '.jsonl':
            # JSON Lines: one JSON object per line
            return [_normalize_record(json.loads(line), source_name=source_name) for line in raw.splitlines() if line.strip()]
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [_normalize_record(item, source_name=source_name) for item in loaded]
        if isinstance(loaded, dict):
            # sometimes a JSON file maps IDs to entries
            return [_normalize_record(item, source_name=source_name) for item in loaded.values()]
    return []

def _normalize_text(value) -> str:
    """Normalize a string to lowercase trimmed form, safe for comparisons."""
    return str(value).strip().lower() if value not in (None, '') else ''

def _extract_skills(value) -> List[str]:
    """Extract a list of skill tokens from various possible input shapes.

    Accepts lists, tuples, comma/semicolon separated strings, and
    attempts to clean common punctuation and list separators.
    """
    if value in (None, '', [], {}):
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    else:
        text = str(value)
        text = text.replace('{', '').replace('}', '').replace('"', '')
        items = re.split(r'[,;|/]', text)
    cleaned = []
    for item in items:
        token = str(item).strip()
        token = token.lstrip('-').strip()
        if token:
            cleaned.append(token)
    return cleaned

def _level_to_rank(level) -> int:
    """Map textual level labels into a coarse numerical rank.

    Uses `LEVEL_RANKS` to interpret strings like 'beginner' or 'advanced'.
    """
    text = _normalize_text(level)
    for key, value in LEVEL_RANKS.items():
        if key in text:
            return value
    return 2 if text else 1

def _level_to_difficulty(level) -> int:
    """Convert a coarse rank into a 1-10 difficulty estimate."""
    rank = _level_to_rank(level)
    return min(10, max(1, rank * 3))

def _parse_duration_to_months(value) -> int:
    """Parse free-form duration strings into months (integer).

    Supports English/Spanish months, weeks and hours. Falls back to
    a sensible minimum of 1 month when parsing fails.
    """
    if value in (None, '', [], {}):
        return 1
    if isinstance(value, (int, float)):
        return max(1, int(round(float(value))))

    text = _normalize_text(value)
    if not text:
        return 1

    # months, e.g. "3-5 months" or "4 months"
    months_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(months?|meses?)', text)
    if months_match:
        start = float(months_match.group(1))
        end = float(months_match.group(2)) if months_match.group(2) else start
        return max(1, int(round((start + end) / 2)))

    # weeks -> approximate to months
    weeks_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(weeks?|semanas?)', text)
    if weeks_match:
        start = float(weeks_match.group(1))
        end = float(weeks_match.group(2)) if weeks_match.group(2) else start
        weeks = (start + end) / 2
        return max(1, int(round(weeks / 4.0)))

    # hours -> approximate by assuming ~20 hours per month of study
    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(hours?|hrs?|h)', text)
    if hours_match:
        start = float(hours_match.group(1))
        end = float(hours_match.group(2)) if hours_match.group(2) else start
        hours = (start + end) / 2
        return max(1, int(round(hours / 20.0)))

    # fallback: extract digits and use them as months
    digits = ''.join(ch for ch in text if ch.isdigit() or ch == '.')
    if digits:
        try:
            return max(1, int(round(float(digits))))
        except ValueError:
            pass
    return 1

def _infer_category_from_text(text: str) -> str:
    """Return a guessed category name based on keyword matching."""
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return category
    return 'General'

def _course_search_blob(course: Dict) -> str:
    """Build a searchable lower-cased text blob for a course.

    Useful for keyword matching and scoring functions.
    """
    parts = [
        course.get('name', '') or course.get('nombre', ''),
        course.get('category', '') or course.get('categoria', ''),
        course.get('type', '') or course.get('tipo', ''),
        course.get('source', ''),
        ' '.join(course.get('skills', [])) if isinstance(course.get('skills'), list) else course.get('skills', ''),
    ]
    return ' '.join(str(part) for part in parts if part).lower()

def _infer_category(record: Dict, fallback: str = 'General') -> str:
    """Infer a course category using available fields and fallbacks.

    Checks multiple metadata fields, then falls back to title+skills.
    """
    for field in ('category', 'categoria', 'subject', 'institution', 'partner', 'topic', 'domain'):
        candidate = record.get(field)
        if candidate not in (None, ''):
            inferred = _infer_category_from_text(str(candidate))
            if inferred != 'General':
                return inferred

    title = record.get('name') or record.get('nombre', '')
    skills = ' '.join(record.get('skills', [])) if isinstance(record.get('skills'), list) else ''
    inferred = _infer_category_from_text(f'{title} {skills}')
    return inferred if inferred != 'General' else fallback

def _score_goal_match(course: Dict, goal_text: str, goal_keywords: Set[str]) -> float:
    """Score how well a course matches a goal using text heuristics.

    Higher weight is given to exact matches in title and helpful keywords
    appearing in the searchable course blob.
    """
    blob = _course_search_blob(course)
    score = 0.0
    title = _normalize_text(course.get('name', '') or course.get('nombre', ''))

    # strong title match
    if goal_text and goal_text in title:
        score += 8.0
    # presence in broader blob
    if goal_text and goal_text in blob:
        score += 6.0

    # keyword-based scoring
    for keyword in goal_keywords:
        if keyword and keyword in blob:
            score += 1.0
        elif keyword and keyword in title:
            score += 1.5

    # small boost for foundational-sounding courses
    if any(term in blob for term in ('introduction', 'fundamentals', 'bootcamp', 'basics', 'learn')):
        score += 0.4

    return score

def _resolve_goal_targets(idx: Dict[str, Dict], goal: str, top_k: int = 4) -> List[str]:
    """Map a free-form goal into a short list of candidate course targets.

    Uses exact name matching first, then falls back to keyword scoring.
    """
    goal_text = _normalize_text(goal)
    if not goal_text:
        return []

    exact_match = next((name for name in idx if _normalize_text(name) == goal_text), None)
    if exact_match:
        return [exact_match]

    # build keyword set from the goal text and role keyword hints
    keywords = set(re.findall(r'[a-z0-9áéíóúñ+#]+', goal_text))
    keywords.update(ROLE_KEYWORDS.get(goal, []))
    if goal_text in ROLE_KEYWORDS:
        keywords.update(ROLE_KEYWORDS[goal_text])

    scored = []
    for name, course in idx.items():
        score = _score_goal_match(course, goal_text, keywords)
        if score > 0:
            scored.append((score, name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:top_k]]

def _infer_prereq_candidates(record: Dict, all_records: List[Dict], idx: Dict[str, Dict]) -> List[str]:
    """Heuristic to propose short list of prerequisite candidate names.

    Uses rule-based triggers to prefer foundational or shorter courses
    that cover common prerequisite topics for the record.
    """
    title_blob = _course_search_blob(record)
    current_rank = _level_to_rank(record.get('level') or record.get('type') or record.get('tipo') or record.get('difficulty') or record.get('dificultad'))

    # skip suggestion for beginner-level content
    if current_rank <= 1 or any(marker in title_blob for marker in ('introduction', 'fundamentals', 'basics', 'bootcamp', 'for everyone', 'beginner', 'getting started')):
        return []

    # topic -> recommended prerequisite term mappings
    topic_rules = [
        (['ai engineering', 'ai engineer', 'mlops'], ['python', 'statistics', 'data science', 'machine learning']),
        (['machine learning', 'deep learning', 'artificial intelligence', 'generative ai', 'nlp'], ['python', 'statistics', 'data science']),
        (['data science', 'data analysis', 'analytics', 'business intelligence'], ['python', 'sql', 'statistics', 'excel']),
        (['web development', 'frontend', 'react', 'javascript', 'html', 'css'], ['javascript', 'programming', 'web']),
        (['backend', 'api', 'server'], ['programming', 'sql', 'database', 'python']),
        (['cloud', 'devops', 'kubernetes', 'docker', 'aws', 'azure'], ['linux', 'programming', 'cloud']),
        (['cybersecurity', 'security', 'cryptography', 'network'], ['network', 'linux', 'programming']),
        (['project management', 'product management', 'leadership'], ['communication', 'planning', 'management']),
        (['design', 'ux', 'ui', 'graphic', 'illustration'], ['design', 'creativity', 'tools']),
        (['python'], ['programming']),
        (['sql'], ['database', 'data']),
    ]

    wanted_terms: List[str] = []
    for trigger_terms, prereq_terms in topic_rules:
        if any(term in title_blob for term in trigger_terms):
            wanted_terms.extend(prereq_terms)

    if not wanted_terms:
        # default to general foundations
        wanted_terms.extend(['introduction', 'fundamentals', 'basics'])

    scored_candidates: List[Tuple[float, str]] = []
    for candidate_name, candidate in idx.items():
        if candidate_name == (record.get('name') or record.get('nombre')):
            continue

        candidate_blob = _course_search_blob(candidate)
        candidate_rank = _level_to_rank(candidate.get('level') or candidate.get('type') or candidate.get('tipo') or candidate.get('difficulty') or candidate.get('dificultad'))
        score = 0.0

        # prefer similar-or-easier rank candidates
        if candidate_rank <= current_rank:
            score += 1.0
        # same category is slightly preferred
        if (candidate.get('category') or candidate.get('categoria')) == (record.get('category') or record.get('categoria')):
            score += 0.75
        wanted_match = any(term in candidate_blob for term in wanted_terms)
        foundation_match = any(term in candidate_blob for term in ('introduction', 'fundamentals', 'basics', 'bootcamp', 'learn'))
        if wanted_match:
            score += 2.5
        if foundation_match:
            score += 0.5
        # prefer shorter or equal-length courses as prerequisites
        if candidate.get('duration_months', candidate.get('duracion_meses', 0)) <= max(1, record.get('duration_months', record.get('duracion_meses', 1))):
            score += 0.25

        # filter out candidates that don't match any desired terms
        if wanted_terms and not wanted_match and not foundation_match:
            continue

        if score > 0:
            scored_candidates.append((score, candidate_name))

    scored_candidates.sort(key=lambda item: (-item[0], idx[item[1]].get('duration_months', idx[item[1]].get('duracion_meses', 0)), item[1]))
    return [name for _, name in scored_candidates[:2]]

def _finalize_records(records: List[Dict]) -> List[Dict]:
    """Normalize raw records and optionally infer missing prerequisites.

    Produces an index of canonical course objects and, for moderate-sized
    datasets, attempts to infer prerequisites to populate the graph.
    """
    normalized = []
    for record in records:
        candidate = _normalize_record(record, source_name=str(record.get('platform', '')))
        if candidate.get('name') or candidate.get('nombre'):
            normalized.append(candidate)

    idx = index_courses(normalized)
    # For large datasets skip expensive inference for performance
    if len(normalized) <= 2000:
        inferred = _infer_missing_prerequisites(normalized, idx)
        for name, prereqs in inferred.items():
            idx.setdefault(name, {})['prerequisites'] = prereqs

    return list(idx.values())

def _infer_missing_prerequisites(records: List[Dict], idx: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Infer missing prerequisites for records grouped by category.

    Builds helpful candidate lists by category and returns a mapping of
    course name -> list of inferred prerequisite names.
    """
    inferred: Dict[str, List[str]] = {}
    by_category: Dict[str, List[Dict]] = {}

    for record in records:
        category = record.get('category') or record.get('categoria') or 'General'
        by_category.setdefault(category, []).append(record)

    # sort each category by difficulty/duration to help pick sensible lower-level courses
    for category_records in by_category.values():
        category_records.sort(key=lambda item: (
            item.get('difficulty', item.get('dificultad', 0)),
            item.get('duration_months', item.get('duracion_meses', 0)),
            item.get('name', item.get('nombre', '')),
        ))

    for record in records:
        # skip if prerequisites already specified in source data
        if record.get('prerequisites') or record.get('prerequisitos'):
            continue
        prereqs = _infer_prereqs_for_record(record, records, idx, by_category)
        if prereqs:
            inferred[record.get('name', record.get('nombre'))] = prereqs

    return inferred

def _infer_prereqs_for_record(record: Dict, records: List[Dict], idx: Dict[str, Dict], by_category: Dict[str, List[Dict]]) -> List[str]:
    """Infer prerequisites for a single record by trying rules then fallbacks."""
    del by_category
    prereqs = _infer_prereq_candidates(record, records, idx)
    if prereqs:
        return prereqs

    # fallback: pick a lower-difficulty course from the same category
    category = record.get('category') or record.get('categoria') or 'General'
    difficulty = record.get('difficulty', record.get('dificultad', 0))
    same_category = [item for item in records if (item.get('category') or item.get('categoria')) == category and (item.get('name') or item.get('nombre')) != (record.get('name') or record.get('nombre'))]
    lower_difficulty = [item for item in same_category if (item.get('difficulty', item.get('dificultad', 0)) < difficulty)]
    if lower_difficulty:
        lower_difficulty.sort(key=lambda item: (
            abs(difficulty - item.get('difficulty', item.get('dificultad', 0))),
            item.get('duration_months', item.get('duracion_meses', 0)),
            item.get('name', item.get('nombre', '')),
        ))
        return [lower_difficulty[0].get('name', lower_difficulty[0].get('nombre'))]

    return []

def _build_goal_prereq_map(goal: str, idx: Dict[str, Dict], max_depth: int = 3) -> Dict[str, Set[str]]:
    """Build a subgraph of prerequisite links for a goal up to `max_depth`.

    The returned dict maps course -> set(prereq_course_names).
    """
    subgraph: Dict[str, Set[str]] = {}
    visited: Set[str] = set()

    def visit(course_name: str, depth: int = 0) -> None:
        # stop when we've either seen this node or reached maximum recursion depth
        if course_name in visited or depth > max_depth:
            return
        visited.add(course_name)

        course = idx.get(course_name)
        if not course:
            return

        # prefer explicit prerequisites, otherwise attempt to infer
        prereqs = set(course.get('prerequisites', course.get('prerequisitos', [])))
        if not prereqs:
            prereqs = set(_infer_prereq_candidates(course, list(idx.values()), idx))

        subgraph[course_name] = prereqs
        for prereq in prereqs:
            if prereq in idx:
                visit(prereq, depth + 1)

    visit(goal)
    return subgraph

def _normalize_record(record: Dict, source_name: str = '') -> Dict:
    """Turn a raw record (various field names) into a canonical course dict.

    Handles Spanish/English field name variants, normalizes types and
    infers a cleaned category. The output is stable for indexing.
    """
    source = {str(key).strip().lower(): value for key, value in dict(record).items()}

    # Accept a wider set of possible field names coming from different CSVs
    nombre = _first_present(source, ['nombre', 'name', 'course', 'course_name', 'course title', 'title', 'course_title'])
    prerequisitos = _parse_prerequisites(_first_present(source, ['prerequisitos', 'prerequisites', 'prereq', 'required_skills', 'requirements']))
    duracion = _parse_duration_to_months(_first_present(source, ['duracion_meses', 'duration_months', 'duration', 'months', 'course_duration']))
    raw_level = _first_present(source, ['nivel', 'level', 'difficulty_level'])
    dificultad = _coerce_number(_first_present(source, ['dificultad', 'difficulty', 'rating']), default=_level_to_difficulty(raw_level))
    categoria = _first_present(source, ['categoria', 'category', 'subject', 'topic', 'domain']) or 'General'
    tipo = _first_present(source, ['tipo', 'type', 'course_type']) or 'curso'
    coste = _coerce_number(_first_present(source, ['coste_usd', 'cost', 'price', 'fee', 'course_fee']), default=0)
    skills = _extract_skills(_first_present(source, ['skills', 'associatedskills', 'skill', 'tags', 'topics']))
    source_label = _first_present(source, ['partner', 'institution', 'source', 'provider', 'instructor']) or ''
    link = _first_present(source, ['link', 'url', 'course_url', 'href']) or ''

    if isinstance(nombre, str):
        nombre = nombre.strip()

    enriched_category = _infer_category({
        'name': nombre or '',
        'categoria': categoria,
        'subject': _first_present(source, ['subject']),
        'category': _first_present(source, ['category']),
        'institution': _first_present(source, ['institution']),
        'partner': _first_present(source, ['partner']),
        'skills': skills,
    }, fallback=categoria or 'General')

    return {
        'name': nombre,
        'prerequisites': prerequisitos,
        'duration_months': int(duracion),
        'difficulty': int(dificultad),
        'category': enriched_category,
        'type': tipo,
        'cost_usd': int(coste),
        'level': raw_level or '',
        'skills': skills,
        'source': source_label,
        'link': link,
        'platform': source_name.lower(),
    }

def _first_present(source: Dict[str, object], candidates: List[str]):
    """Return the first non-empty candidate value from a mapping."""
    for candidate in candidates:
        if candidate in source and source[candidate] not in (None, ''):
            return source[candidate]
    return None

def _parse_prerequisites(value) -> List[str]:
    """Normalize various prerequisite representations into a list of names."""
    if value in (None, '', [], {}):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().lower() not in {'none', 'none.', 'null', 'nan'}]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().lower() not in {'none', 'none.', 'null', 'nan'}]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(';', ',').split(',')]
        return [part for part in parts if part and part.lower() not in {'none', 'none.', 'null', 'nan'}]
    return [str(value).strip()]

def _coerce_number(value, default: int = 0) -> int:
    """Safely coerce various number-like inputs into an integer.

    Interprets words like 'free' and strips punctuation when parsing.
    """
    if value in (None, '', [], {}):
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in {'free', 'gratis', 'sin coste'}:
            return 0
        digits = ''.join(ch for ch in text if ch.isdigit() or ch in {'.', '-'})
        if digits:
            try:
                return int(round(float(digits)))
            except ValueError:
                return default
    return default

def _course_richness_score(course: Dict) -> int:
    """A simple heuristic indicating how informative a course record is.

    Higher scores indicate records with more metadata, skills and
    prerequisites which are preferred when deduplicating.
    """
    score = len(_extract_skills(course.get('skills'))) * 4
    score += len(_parse_prerequisites(course.get('prerequisites') or course.get('prerequisitos'))) * 2

    spanish_fallback = {'name': 'nombre', 'category': 'categoria', 'type': 'tipo'}
    for field in ('name', 'category', 'type', 'level', 'source', 'link', 'platform'):
        val = course.get(field)
        if val in (None, '', [], {}):
            # try Spanish fallback for a subset of fields
            if field in spanish_fallback:
                val = course.get(spanish_fallback[field])
        if val not in (None, '', [], {}):
            score += 1

    return score

def index_courses(courses: List[Dict]) -> Dict[str, Dict]:
    """Build a name -> course mapping, preferring richer records.

    When multiple records share the same name, the richer one wins.
    """
    indexed: Dict[str, Dict] = {}

    for course in courses:
        name = course.get('name') or course.get('nombre')
        if not name:
            continue

        existing = indexed.get(name)
        if existing is None or _course_richness_score(course) > _course_richness_score(existing):
            indexed[name] = course

    return indexed

def build_required_closure(goal: str, prereq_map: Dict[str, Set[str]]) -> Set[str]:
    """Return the transitive closure of prerequisites required for `goal`.

    Walks the prereq_map recursively collecting all prerequisite nodes.
    """
    required: Set[str] = set()

    def visit(node: str):
        for prereq in prereq_map.get(node, set()):
            if prereq not in required:
                required.add(prereq)
                visit(prereq)

    visit(goal)
    return required

def generate_paths(courses: List[Dict], initial_skills: List[str], goal: str,
                   max_paths: int = 3, max_depth: int = 12,
                   avoid_categories: Set[str] = None,
                   user_prefs: Optional[Dict] = None,
                   criteria_names: Optional[List[str]] = None) -> List[Dict]:
    # SECCION Path Generation & Search -------------------------------------

    """Generate a set of candidate learning paths to reach `goal`.

    Produces multiple solutions according to different user-selected
    criteria profiles (fastest, cheapest, balanced, etc.).
    """

    idx = index_courses(courses)
    category = {name: _get_category(c) for name, c in idx.items()}
    goal_targets = _resolve_goal_targets(idx, goal)
    if not goal_targets and goal in idx:
        goal_targets = [goal]
    if not goal_targets:
        return []

    criteria_profiles = {
        'rapida': {
            'name': 'rapida',
            'label': 'Fastest path',
            'duration_weight': 1.2,
            'difficulty_weight': 0.20,
            'cost_weight': 0.05,
            'category_bias': {'Datos': -0.25, 'Web': -0.2, 'Fundamentos': -0.15},
        },
        'economica': {
            'name': 'economica',
            'label': 'Cheapest path',
            'duration_weight': 0.75,
            'difficulty_weight': 0.20,
            'cost_weight': 0.12,
            'category_bias': {'Cloud': -0.15, 'Web': -0.1, 'Fundamentos': -0.1},
        },
        'balanceada': {
            'name': 'balanceada',
            'label': 'Balanced path',
            'duration_weight': 1.0,
            'difficulty_weight': 0.45,
            'cost_weight': 0.08,
            'category_bias': {'IA': -0.2, 'CS': -0.1, 'Fundamentos': -0.1},
        },
        'tecnica': {
            'name': 'tecnica',
            'label': 'Most technical path',
            'duration_weight': 0.9,
            'difficulty_weight': 0.7,
            'cost_weight': 0.05,
            'category_bias': {'IA': -0.45, 'Matemáticas': -0.35, 'DevOps': -0.15},
        },
        'infra': {
            'name': 'infra',
            'label': 'Cloud / infrastructure path',
            'duration_weight': 1.0,
            'difficulty_weight': 0.35,
            'cost_weight': 0.05,
            'category_bias': {'Cloud': -0.45, 'DevOps': -0.4, 'Web': -0.15},
        },
    }

    selected_criteria = criteria_names or ['rapida', 'economica', 'balanceada']
    profiles = [criteria_profiles[name] for name in selected_criteria if name in criteria_profiles]
    if not profiles:
        profiles = [criteria_profiles['rapida'], criteria_profiles['economica'], criteria_profiles['balanceada']]

    results: List[Dict] = []
    seen_signatures: Set[Tuple[str, ...]] = set()

    for profile in profiles:
        if len(results) >= max_paths:
            break

        for target in goal_targets:
            if len(results) >= max_paths:
                break

            target_prereq_map = _build_goal_prereq_map(target, idx)
            required_for_goal = build_required_closure(target, target_prereq_map) | {target}

            # If user lacks Python, try to ensure a Python course is included as an initial required step
            if user_prefs and user_prefs.get('knows_python') is False:
                # find a plausible Python course in the index
                python_course = None
                # prefer a course with 'python' in its title
                for name, c in idx.items():
                    if 'python' in (name or '').lower():
                        python_course = name
                        break
                # fallback: any course that mentions python in the blob
                if python_course is None:
                    for name, c in idx.items():
                        blob = _course_search_blob(c)
                        if 'python' in blob:
                            python_course = name
                            break
                if python_course:
                    # enforce python_course as a prerequisite for the chosen target so it's included first
                    target_prereq_map.setdefault(target, set()).add(python_course)
                    required_for_goal.add(python_course)
            solution = search_best_path(
                idx=idx,
                prereq_map=target_prereq_map,
                category=category,
                initial_skills=initial_skills,
                goal=target,
                required_for_goal=required_for_goal,
                avoid_categories=avoid_categories,
                max_depth=max_depth,
                profile=profile,
            )

            if not solution:
                continue

            signature = tuple(solution['path'])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            solution['objective'] = goal
            solution['target_course'] = target
            solution['criterion'] = profile.get('label', profile['name'])
            results.append(solution)

    # If some profiles collapsed to the same solution, try a few more greedy variations.
    if len(results) < max_paths:
        backup_profiles = [
            {
                'name': 'entrada_rapida',
                'duration_weight': 1.15,
                'difficulty_weight': 0.1,
                'cost_weight': 0.05,
                'category_bias': {'Data': -0.2, 'Datos': -0.3, 'Web': -0.15},
            },
            {
                'name': 'ml_practico',
                'duration_weight': 0.95,
                'difficulty_weight': 0.55,
                'cost_weight': 0.05,
                'category_bias': {'IA': -0.3, 'DevOps': -0.2, 'Cloud': -0.1},
            },
        ]
        for profile in backup_profiles:
            if len(results) >= max_paths:
                break
            for target in goal_targets:
                if len(results) >= max_paths:
                    break
                target_prereq_map = _build_goal_prereq_map(target, idx)
                required_for_goal = build_required_closure(target, target_prereq_map) | {target}
                solution = search_best_path(
                    idx=idx,
                    prereq_map=target_prereq_map,
                    category=category,
                    initial_skills=initial_skills,
                    goal=target,
                    required_for_goal=required_for_goal,
                    avoid_categories=avoid_categories,
                    max_depth=max_depth,
                    profile=profile,
                )
                if not solution:
                    continue
                signature = tuple(solution['path'])
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                solution['objective'] = goal
                solution['target_course'] = target
                solution['criterion'] = profile.get('label', profile['name'])
                results.append(solution)

    return results

def search_best_path(
    idx: Dict[str, Dict],
    prereq_map: Dict[str, Set[str]],
    category: Dict[str, str],
    initial_skills: List[str],
    goal: str,
    required_for_goal: Set[str],
    avoid_categories: Optional[Set[str]],
    max_depth: int,
    profile: Dict,
) -> Optional[Dict]:
    """Search for the best path to `goal` using an A*-like search.

    The algorithm expands states represented by sets of acquired skills
    and courses, scoring each candidate by `node_cost` + heuristic.
    """
    start_acquired = set(initial_skills)
    start_path: List[str] = []
    start_state = frozenset(start_acquired)
    # priority queue entries: (f_score, g_score, counter, state, path)
    queue: List[Tuple[float, float, int, frozenset, List[str]]] = []
    counter = 0
    heapq.heappush(queue, (heuristic(start_acquired, goal, idx, prereq_map), 0.0, counter, start_state, start_path))
    best_seen: Dict[frozenset, float] = {start_state: 0.0}

    while queue:
        f_score, g_score, _, state, path = heapq.heappop(queue)
        acquired = set(state)

        # success: goal already acquired
        if goal in acquired:
            return {
                'path': path,
                'metrics': score_path(path, idx),
                'profile': profile['name'],
            }

        # depth guard
        if len(path) >= max_depth:
            continue

        # compute available candidate courses that have prerequisites satisfied
        available = []
        for course in idx.keys():
            if course in acquired:
                continue
            if avoid_categories and category.get(course) in avoid_categories:
                continue
            reqs = prereq_map.get(course, set())
            if not reqs.issubset(acquired):
                continue
            # only consider courses that are required for the chosen goal
            if course not in required_for_goal and course != goal:
                continue
            available.append(course)

        # sort candidates by heuristic node cost, duration, then name
        available.sort(key=lambda name: (
            node_cost(name, idx, profile),
            _get_duration(idx[name]),
            name,
        ))

        for course in available:
            new_acquired = set(acquired)
            new_acquired.add(course)
            new_state = frozenset(new_acquired)
            new_path = path + [course]
            new_g = g_score + node_cost(course, idx, profile)
            if new_state in best_seen and best_seen[new_state] <= new_g:
                continue
            best_seen[new_state] = new_g
            counter += 1
            new_h = heuristic(new_acquired, goal, idx, prereq_map)
            heapq.heappush(queue, (new_g + new_h, new_g, counter, new_state, new_path))

    return None

def score_path(path: List[str], idx: Dict[str, Dict]) -> Dict:
    """Compute simple aggregate metrics for a path of course names."""
    total_months = 0
    total_cost = 0
    difficulties = []
    for node in path:
        info = idx.get(node, {})
        total_months += _get_duration(info)
        total_cost += _get_cost(info)
        d = _get_difficulty(info)
        if d is not None:
            difficulties.append(d)
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
    return {
        'total_months': total_months,
        'total_cost': total_cost,
        'avg_difficulty': round(avg_difficulty, 2),
        'steps': len(path)
    }

def node_cost(name: str, idx: Dict[str, Dict], profile: Dict) -> float:
    """Compute a scalar cost for selecting a single course under a profile."""
    info = idx.get(name, {})
    category = _get_category(info)
    category_bias = profile.get('category_bias', {}).get(category, 0.0)
    return (
        profile.get('duration_weight', 1.0) * _get_duration(info)
        + profile.get('difficulty_weight', 0.35) * _get_difficulty(info)
        + profile.get('cost_weight', 0.05) * _get_cost(info)
        + category_bias
    )

def heuristic(acquired: Set[str], goal: str, idx: Dict[str, Dict], prereq_map: Dict[str, Set[str]]) -> float:
    """Admissible heuristic estimating remaining duration to reach goal.

    Sums durations of missing required nodes as a simple lower bound.
    """
    required = build_required_closure(goal, prereq_map) | {goal}
    missing = [node for node in required if node not in acquired]
    return sum(_get_duration(idx.get(node, {})) for node in missing)


if __name__ == '__main__':
    # Quick local smoke test
    courses = load_courses()
    paths = generate_paths(courses, ['Python'], 'AI Engineer', max_paths=3)
    for p in paths:
        print(p)
