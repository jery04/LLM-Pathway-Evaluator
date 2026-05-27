import csv
import heapq
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


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


def _normalize_text(value) -> str:
    return str(value).strip().lower() if value not in (None, '') else ''


def _extract_skills(value) -> List[str]:
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
    text = _normalize_text(level)
    for key, value in LEVEL_RANKS.items():
        if key in text:
            return value
    return 2 if text else 1


def _level_to_difficulty(level) -> int:
    rank = _level_to_rank(level)
    return min(10, max(1, rank * 3))


def _parse_duration_to_months(value) -> int:
    if value in (None, '', [], {}):
        return 1
    if isinstance(value, (int, float)):
        return max(1, int(round(float(value))))

    text = _normalize_text(value)
    if not text:
        return 1

    months_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(months?|meses?)', text)
    if months_match:
        start = float(months_match.group(1))
        end = float(months_match.group(2)) if months_match.group(2) else start
        return max(1, int(round((start + end) / 2)))

    weeks_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(weeks?|semanas?)', text)
    if weeks_match:
        start = float(weeks_match.group(1))
        end = float(weeks_match.group(2)) if weeks_match.group(2) else start
        weeks = (start + end) / 2
        return max(1, int(round(weeks / 4.0)))

    hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(hours?|hrs?|h)', text)
    if hours_match:
        start = float(hours_match.group(1))
        end = float(hours_match.group(2)) if hours_match.group(2) else start
        hours = (start + end) / 2
        return max(1, int(round(hours / 20.0)))

    digits = ''.join(ch for ch in text if ch.isdigit() or ch == '.')
    if digits:
        try:
            return max(1, int(round(float(digits))))
        except ValueError:
            pass
    return 1


def _infer_category_from_text(text: str) -> str:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return category
    return 'General'


def _course_search_blob(course: Dict) -> str:
    parts = [
        course.get('nombre', ''),
        course.get('categoria', ''),
        course.get('tipo', ''),
        course.get('source', ''),
        ' '.join(course.get('skills', [])) if isinstance(course.get('skills'), list) else course.get('skills', ''),
    ]
    return ' '.join(str(part) for part in parts if part).lower()


def _infer_category(record: Dict, fallback: str = 'General') -> str:
    for field in ('categoria', 'subject', 'category', 'institution', 'partner', 'topic', 'domain'):
        candidate = record.get(field)
        if candidate not in (None, ''):
            inferred = _infer_category_from_text(str(candidate))
            if inferred != 'General':
                return inferred

    title = record.get('nombre', '')
    skills = ' '.join(record.get('skills', [])) if isinstance(record.get('skills'), list) else ''
    inferred = _infer_category_from_text(f'{title} {skills}')
    return inferred if inferred != 'General' else fallback


def _score_goal_match(course: Dict, goal_text: str, goal_keywords: Set[str]) -> float:
    blob = _course_search_blob(course)
    score = 0.0
    title = _normalize_text(course.get('nombre', ''))

    if goal_text and goal_text in title:
        score += 8.0
    if goal_text and goal_text in blob:
        score += 6.0

    for keyword in goal_keywords:
        if keyword and keyword in blob:
            score += 1.0
        elif keyword and keyword in title:
            score += 1.5

    # Encourage courses that look like foundations for a role.
    if any(term in blob for term in ('introduction', 'fundamentals', 'bootcamp', 'basics', 'learn')):
        score += 0.4

    return score


def _resolve_goal_targets(idx: Dict[str, Dict], goal: str, top_k: int = 4) -> List[str]:
    goal_text = _normalize_text(goal)
    if not goal_text:
        return []

    exact_match = next((name for name in idx if _normalize_text(name) == goal_text), None)
    if exact_match:
        return [exact_match]

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
    title_blob = _course_search_blob(record)
    current_rank = _level_to_rank(record.get('level') or record.get('tipo') or record.get('dificultad'))

    if current_rank <= 1 or any(marker in title_blob for marker in ('introduction', 'fundamentals', 'basics', 'bootcamp', 'for everyone', 'beginner', 'getting started')):
        return []

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
        wanted_terms.extend(['introduction', 'fundamentals', 'basics'])

    scored_candidates: List[Tuple[float, str]] = []
    for candidate_name, candidate in idx.items():
        if candidate_name == record.get('nombre'):
            continue

        candidate_blob = _course_search_blob(candidate)
        candidate_rank = _level_to_rank(candidate.get('level') or candidate.get('tipo') or candidate.get('dificultad'))
        score = 0.0

        if candidate_rank <= current_rank:
            score += 1.0
        if candidate.get('categoria') == record.get('categoria'):
            score += 0.75
        wanted_match = any(term in candidate_blob for term in wanted_terms)
        foundation_match = any(term in candidate_blob for term in ('introduction', 'fundamentals', 'basics', 'bootcamp', 'learn'))
        if wanted_match:
            score += 2.5
        if foundation_match:
            score += 0.5
        if candidate.get('duracion_meses', 0) <= max(1, record.get('duracion_meses', 1)):
            score += 0.25

        if wanted_terms and not wanted_match and not foundation_match:
            continue

        if score > 0:
            scored_candidates.append((score, candidate_name))

    scored_candidates.sort(key=lambda item: (-item[0], idx[item[1]].get('duracion_meses', 0), item[1]))
    return [name for _, name in scored_candidates[:2]]


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
        prereqs = _infer_prereqs_for_record(record, records, idx, by_category)
        if prereqs:
            inferred[record['nombre']] = prereqs

    return inferred


def _infer_prereqs_for_record(record: Dict, records: List[Dict], idx: Dict[str, Dict], by_category: Dict[str, List[Dict]]) -> List[str]:
    del by_category
    prereqs = _infer_prereq_candidates(record, records, idx)
    if prereqs:
        return prereqs

    category = record.get('categoria') or 'General'
    difficulty = record.get('dificultad', 0)
    same_category = [item for item in records if item.get('categoria') == category and item.get('nombre') != record.get('nombre')]
    lower_difficulty = [item for item in same_category if item.get('dificultad', 0) < difficulty]
    if lower_difficulty:
        lower_difficulty.sort(key=lambda item: (
            abs(difficulty - item.get('dificultad', 0)),
            item.get('duracion_meses', 0),
            item.get('nombre', ''),
        ))
        return [lower_difficulty[0]['nombre']]

    return []


def _build_goal_prereq_map(goal: str, idx: Dict[str, Dict], max_depth: int = 3) -> Dict[str, Set[str]]:
    subgraph: Dict[str, Set[str]] = {}
    visited: Set[str] = set()

    def visit(course_name: str, depth: int = 0) -> None:
        if course_name in visited or depth > max_depth:
            return
        visited.add(course_name)

        course = idx.get(course_name)
        if not course:
            return

        prereqs = set(course.get('prerequisitos', []))
        if not prereqs:
            prereqs = set(_infer_prereq_candidates(course, list(idx.values()), idx))

        subgraph[course_name] = prereqs
        for prereq in prereqs:
            if prereq in idx:
                visit(prereq, depth + 1)

    visit(goal)
    return subgraph


def _normalize_record(record: Dict) -> Dict:
    source = {str(key).strip().lower(): value for key, value in dict(record).items()}

    # Accept a wider set of possible field names coming from different CSVs
    nombre = _first_present(source, ['nombre', 'name', 'course', 'course_name', 'course title', 'course title', 'title', 'course_title'])
    prerequisitos = _parse_prerequisites(_first_present(source, ['prerequisitos', 'prerequisites', 'prereq', 'required_skills', 'requirements']))
    duracion = _parse_duration_to_months(_first_present(source, ['duracion_meses', 'duration_months', 'duration', 'months', 'course_duration']))
    raw_level = _first_present(source, ['nivel', 'level', 'difficulty_level'])
    dificultad = _coerce_number(_first_present(source, ['dificultad', 'difficulty', 'rating']), default=_level_to_difficulty(raw_level))
    categoria = _first_present(source, ['categoria', 'category', 'subject', 'topic', 'domain']) or 'General'
    tipo = _first_present(source, ['tipo', 'type', 'course_type']) or 'curso'
    coste = _coerce_number(_first_present(source, ['coste_usd', 'cost', 'price', 'fee', 'course_fee']), default=0)
    skills = _extract_skills(_first_present(source, ['skills', 'associatedskills', 'skill', 'tags', 'topics']))
    source_label = _first_present(source, ['partner', 'institution', 'source', 'provider', 'instructor']) or ''

    if isinstance(nombre, str):
        nombre = nombre.strip()

    enriched_category = _infer_category({
        'nombre': nombre or '',
        'categoria': categoria,
        'subject': _first_present(source, ['subject']),
        'category': _first_present(source, ['category']),
        'institution': _first_present(source, ['institution']),
        'partner': _first_present(source, ['partner']),
        'skills': skills,
    }, fallback=categoria or 'General')

    return {
        'nombre': nombre,
        'prerequisitos': prerequisitos,
        'duracion_meses': int(duracion),
        'dificultad': int(dificultad),
        'categoria': enriched_category,
        'tipo': tipo,
        'coste_USD': int(coste),
        'level': raw_level or '',
        'skills': skills,
        'source': source_label,
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
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().lower() not in {'none', 'none.', 'null', 'nan'}]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().lower() not in {'none', 'none.', 'null', 'nan'}]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(';', ',').split(',')]
        return [part for part in parts if part and part.lower() not in {'none', 'none.', 'null', 'nan'}]
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
    category = {name: c.get('categoria') for name, c in idx.items()}
    goal_targets = _resolve_goal_targets(idx, goal)
    if not goal_targets and goal in idx:
        goal_targets = [goal]
    if not goal_targets:
        return []

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
