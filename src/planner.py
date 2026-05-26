import csv
import heapq
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


KAGGLE_DATASET_ID = "everydaycodings/multi-platform-online-courses-dataset"
KAGGLE_DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "kagglehub"


def load_courses(path: Optional[str] = None) -> List[Dict]:
    if path:
        candidate = Path(path)
        if candidate.exists():
            return _finalize_records(_load_records_from_path(candidate))

    # First try to load any local data shipped with the repository (data/*.csv, json, jsonl).
    local_data_dir = Path(__file__).resolve().parent.parent / "data"
    if local_data_dir.exists():
        local_records = _load_records_from_path(local_data_dir)
        if local_records:
            return _finalize_records(local_records)

    # Fall back to attempting to download via kagglehub if local data isn't available.
    downloaded = _load_kaggle_dataset(KAGGLE_DATASET_ID)

    return _finalize_records(downloaded)


def _load_kaggle_dataset(dataset_id: str) -> List[Dict]:
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
    if path.is_file():
        return _load_records_from_file(path)

    records: List[Dict] = []
    for candidate in sorted(path.rglob('*')):
        if candidate.suffix.lower() not in {'.csv', '.json', '.jsonl'}:
            continue
        records.extend(_load_records_from_file(candidate))
    return records


def _load_records_from_file(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    with open(path, 'r', encoding='utf-8') as handle:
        raw = handle.read().strip()
        if not raw:
            return []
        if suffix == '.jsonl':
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return list(loaded.values())
    return []


def _finalize_records(records: List[Dict]) -> List[Dict]:
    normalized = []
    for record in records:
        candidate = _normalize_record(record)
        if candidate.get('nombre'):
            normalized.append(candidate)

    idx = index_courses(normalized)
    # For large source datasets, skip expensive prerequisite inference so the
    # app can load quickly from the bundled CSVs. The course graph still works
    # with explicit prerequisites when present in the source data.
    if len(normalized) <= 2000:
        inferred = _infer_missing_prerequisites(normalized, idx)
        for name, prereqs in inferred.items():
            idx[name]['prerequisitos'] = prereqs

    return list(idx.values())


def _infer_missing_prerequisites(records: List[Dict], idx: Dict[str, Dict]) -> Dict[str, List[str]]:
    inferred: Dict[str, List[str]] = {}
    by_category: Dict[str, List[Dict]] = {}

    for record in records:
        category = record.get('categoria') or 'General'
        by_category.setdefault(category, []).append(record)

    for category_records in by_category.values():
        category_records.sort(key=lambda item: (
            item.get('dificultad', 0),
            item.get('duracion_meses', 0),
            item.get('nombre', ''),
        ))

    for record in records:
        if record.get('prerequisitos'):
            continue
        prereqs = _infer_prereqs_for_record(record, by_category)
        if prereqs:
            inferred[record['nombre']] = prereqs

    return inferred


def _infer_prereqs_for_record(record: Dict, by_category: Dict[str, List[Dict]]) -> List[str]:
    name = (record.get('nombre') or '').lower()
    category = record.get('categoria') or 'General'
    difficulty = record.get('dificultad', 0)

    keyword_rules = [
        (['python'], []),
        (['sql'], ['Python']),
        (['data analysis', 'data analytics', 'analytics'], ['Python', 'SQL']),
        (['statistics', 'statistical'], ['Python']),
        (['linear algebra'], []),
        (['calculus'], []),
        (['data structures'], ['Python']),
        (['algorithms'], ['Data Structures']),
        (['machine learning', 'ml '], ['Python', 'Statistics']),
        (['deep learning'], ['Machine Learning', 'Linear Algebra']),
        (['nlp', 'natural language'], ['Machine Learning']),
        (['mlops'], ['Machine Learning', 'Docker']),
        (['docker'], ['Python']),
        (['kubernetes'], ['Docker']),
        (['cloud fundamentals', 'cloud'], []),
        (['aws'], ['Cloud Fundamentals']),
        (['devops'], ['Docker']),
        (['backend'], ['Python']),
        (['api'], ['Backend Development']),
        (['cybersecurity'], []),
        (['networking'], []),
    ]

    for tokens, prereqs in keyword_rules:
        if any(token in name for token in tokens):
            return prereqs[:]

    same_category = [item for item in by_category.get(category, []) if item.get('nombre') != record.get('nombre')]
    lower_difficulty = [item for item in same_category if item.get('dificultad', 0) < difficulty]
    if lower_difficulty:
        lower_difficulty.sort(key=lambda item: (
            abs(difficulty - item.get('dificultad', 0)),
            item.get('duracion_meses', 0),
            item.get('nombre', ''),
        ))
        return [lower_difficulty[0]['nombre']]

    return []


def _normalize_record(record: Dict) -> Dict:
    source = {str(key).strip().lower(): value for key, value in dict(record).items()}

    # Accept a wider set of possible field names coming from different CSVs
    nombre = _first_present(source, ['nombre', 'name', 'course', 'course_name', 'course title', 'course title', 'title', 'course_title'])
    prerequisitos = _parse_prerequisites(_first_present(source, ['prerequisitos', 'prerequisites', 'prereq', 'required_skills', 'requirements']))
    duracion = _coerce_number(_first_present(source, ['duracion_meses', 'duration_months', 'duration', 'months', 'course_duration']), default=1)
    dificultad = _coerce_number(_first_present(source, ['dificultad', 'difficulty', 'level', 'difficulty_level', 'rating']), default=5)
    categoria = _first_present(source, ['categoria', 'category', 'subject', 'topic', 'domain']) or 'General'
    tipo = _first_present(source, ['tipo', 'type', 'course_type']) or 'curso'
    coste = _coerce_number(_first_present(source, ['coste_usd', 'cost', 'price', 'fee', 'course_fee']), default=0)

    if isinstance(nombre, str):
        nombre = nombre.strip()

    return {
        'nombre': nombre,
        'prerequisitos': prerequisitos,
        'duracion_meses': int(duracion),
        'dificultad': int(dificultad),
        'categoria': categoria,
        'tipo': tipo,
        'coste_USD': int(coste),
    }


def _first_present(source: Dict[str, object], candidates: List[str]):
    for candidate in candidates:
        if candidate in source and source[candidate] not in (None, ''):
            return source[candidate]
    return None


def _parse_prerequisites(value) -> List[str]:
    if value in (None, '', [], {}):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(';', ',').split(',')]
        return [part for part in parts if part]
    return [str(value).strip()]


def _coerce_number(value, default: int = 0) -> int:
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


def index_courses(courses: List[Dict]) -> Dict[str, Dict]:
    return {c['nombre']: c for c in courses}


def build_required_closure(goal: str, prereq_map: Dict[str, Set[str]]) -> Set[str]:
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
                   avoid_categories: Set[str] = None) -> List[Dict]:
    idx = index_courses(courses)
    prereq_map = {name: set(c.get('prerequisitos', [])) for name, c in idx.items()}
    category = {name: c.get('categoria') for name, c in idx.items()}
    required_for_goal = build_required_closure(goal, prereq_map)

    profiles = [
        {
            'name': 'rapida',
            'duration_weight': 1.0,
            'difficulty_weight': 0.25,
            'cost_weight': 0.05,
            'category_bias': {'Datos': -0.35, 'Web': -0.25, 'Fundamentos': -0.15},
        },
        {
            'name': 'balanceada',
            'duration_weight': 1.0,
            'difficulty_weight': 0.45,
            'cost_weight': 0.08,
            'category_bias': {'IA': -0.2, 'CS': -0.1, 'Fundamentos': -0.1},
        },
        {
            'name': 'tecnica',
            'duration_weight': 0.9,
            'difficulty_weight': 0.7,
            'cost_weight': 0.05,
            'category_bias': {'IA': -0.45, 'Matemáticas': -0.35, 'DevOps': -0.15},
        },
        {
            'name': 'infra',
            'duration_weight': 1.0,
            'difficulty_weight': 0.35,
            'cost_weight': 0.05,
            'category_bias': {'Cloud': -0.45, 'DevOps': -0.4, 'Web': -0.15},
        },
    ]

    results: List[Dict] = []
    seen_signatures: Set[Tuple[str, ...]] = set()

    for profile in profiles:
        if len(results) >= max_paths:
            break

        solution = search_best_path(
            idx=idx,
            prereq_map=prereq_map,
            category=category,
            initial_skills=initial_skills,
            goal=goal,
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
            solution = search_best_path(
                idx=idx,
                prereq_map=prereq_map,
                category=category,
                initial_skills=initial_skills,
                goal=goal,
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
    start_acquired = set(initial_skills)
    start_path: List[str] = []
    start_state = frozenset(start_acquired)
    queue: List[Tuple[float, float, int, frozenset, List[str]]] = []
    counter = 0
    heapq.heappush(queue, (heuristic(start_acquired, goal, idx, prereq_map), 0.0, counter, start_state, start_path))
    best_seen: Dict[frozenset, float] = {start_state: 0.0}

    while queue:
        f_score, g_score, _, state, path = heapq.heappop(queue)
        acquired = set(state)

        if goal in acquired:
            return {
                'path': path,
                'metrics': score_path(path, idx),
                'profile': profile['name'],
            }

        if len(path) >= max_depth:
            continue

        available = []
        for course in idx.keys():
            if course in acquired:
                continue
            if avoid_categories and category.get(course) in avoid_categories:
                continue
            reqs = prereq_map.get(course, set())
            if not reqs.issubset(acquired):
                continue
            if course not in required_for_goal and course != goal:
                continue
            available.append(course)

        available.sort(key=lambda name: (
            node_cost(name, idx, profile),
            idx[name].get('duracion_meses', 0),
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
    total_months = 0
    total_cost = 0
    difficulties = []
    for node in path:
        info = idx.get(node, {})
        total_months += info.get('duracion_meses', 0)
        total_cost += info.get('coste_USD', 0)
        d = info.get('dificultad')
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
    info = idx.get(name, {})
    category = info.get('categoria')
    category_bias = profile.get('category_bias', {}).get(category, 0.0)
    return (
        profile.get('duration_weight', 1.0) * info.get('duracion_meses', 0)
        + profile.get('difficulty_weight', 0.35) * info.get('dificultad', 0)
        + profile.get('cost_weight', 0.05) * info.get('coste_USD', 0)
        + category_bias
    )


def heuristic(acquired: Set[str], goal: str, idx: Dict[str, Dict], prereq_map: Dict[str, Set[str]]) -> float:
    required = build_required_closure(goal, prereq_map) | {goal}
    missing = [node for node in required if node not in acquired]
    return sum(idx.get(node, {}).get('duracion_meses', 0) for node in missing)


if __name__ == '__main__':
    # Quick local smoke test
    courses = load_courses()
    paths = generate_paths(courses, ['Python'], 'AI Engineer', max_paths=3)
    for p in paths:
        print(p)
