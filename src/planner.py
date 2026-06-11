"""planner.py - Learning path generator from online course catalogs.

This module builds optimized learning pathways using recursive prerequisite 
resolution, embedding-based similarity matching, and multi-criteria scoring.

Core capabilities:
    - Load courses and embeddings from JSON files
    - Build skill, category and course lookup indexes
    - Infer prerequisite skills for courses via LLM
    - Find courses that teach a given skill (embedding + lexical search)
    - Score and rank courses by cost, duration and difficulty
    - Generate recursive prerequisite trees with alternatives tracking
    - Produce multiple ranked paths from initial skills to target objective

Depends on llm_adapter for embeddings and prerequisite inference.
"""

import json                 # Loads and dumps JSON data structures
from pathlib import Path    # Object‑oriented filesystem path handling
from typing import List, Dict, Set, Optional   # Type hints for common container types
from collections import defaultdict    # Collections for graph data structures (prerequisite DAG)
import math     # Math functions for cosine similarity calculations
from dataclasses import dataclass, field    # Defines structured data classes (CourseNode)

# Import LLM adapter for prerequisite inference and embeddings
from llm_adapter import infer_prerequisites_for_objective, get_text_embedding

# Import STRIPS planner for path refinement using classical AI planning (BFS/A*)
from strips_search import refine_paths_with_planner

# Global thresholds for similarity-based matching
SIMILARITY_THRESHOLD = 0.75
CATEGORY_SIMILARITY_THRESHOLD = 0.6


# =============================================================================
# COURSE HIERARCHY (TREE MODEL)
# =============================================================================

@dataclass
class CourseNode:
    """Node representing a course in the prerequisite tree.
    Stores the course name, satisfied skill, alternatives, unresolved skills and children.
    """
    name: str
    skill: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    children: List["CourseNode"] = field(default_factory=list)

    @staticmethod
    def flatten(node: "CourseNode") -> List[str]:
        """Return a post-order list of course names from the subtree.
        Useful to produce an ordered linear learning path from a tree.
        """
        result = []
        for child in node.children:
            result.extend(CourseNode.flatten(child))
        result.append(node.name)
        return result
    
    @staticmethod
    def print_tree(node: "CourseNode", indent: int = 0) -> None:
        """Print a readable tree of the node and its descendants.
        Shows course name, satisfied skill, alternatives and unresolved skills.
        """
        prefix = " " * (indent * 4)

        # nodo principal
        print(f"{prefix}📘 {node.name}")

        # skill que satisface este nodo (si existe)
        if node.skill:
            print(f"{prefix}   🎯 skill: {node.skill}")

        # alternativas
        if node.alternatives:
            print(f"{prefix}   🔁 alternatives: {node.alternatives}")

        # skills no resueltas
        if node.unresolved:
            print(f"{prefix}   ⚠️ unresolved: {node.unresolved}")

        # hijos
        for child in node.children:
            CourseNode.print_tree(child, indent + 1)


# =====================================================================
# DATA LOADING, EMBEDDING & INDEX BUILDING
# =====================================================================

def load_courses() -> List[Dict]:
    """Load courses from normalized_courses.json.
    
    Returns a list of course dictionaries with fields:
    - name: course title
    - duration_months: estimated duration in months
    - difficulty: difficulty score (1-10)
    - category: course category (e.g., "Data", "Web", "AI", etc.)
    - cost_usd: course cost in USD
    - level: beginner/intermediate/advanced
    - skills: list of skills taught
    - link: course link (may be empty)
    """
    json_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "normalized_courses.json"
    )

    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _load_embeddings_data(experiment: bool = False) -> Optional[Dict[str, List[float]]]:
    """Load embeddings from data/embedding.json as {name: embedding}.

    If `experiment` is True, load from `experiments/embedding.json` instead.
    """
    base_dir = 'experiments' if experiment else 'data'
    embedding_path = (
        Path(__file__).resolve().parent.parent
        / base_dir
        / 'embedding.json'
    )

    try:
        if not embedding_path.exists():
            return None
        with open(embedding_path, encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return None

        embeddings_map = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            name = (item.get('name') or '').strip()
            embedding = item.get('embedding')
            if not name or not isinstance(embedding, list):
                continue

            # Keep first seen embedding for stable behavior.
            if name not in embeddings_map:
                embeddings_map[name] = embedding

        return embeddings_map or None
    except Exception:
        return None

def index_courses(courses: List[Dict]) -> Dict[str, Dict]:
    """Create a lookup index for courses by name (preferring richer records).
    
    Handles duplicate course names by keeping the entry with the most skills.
    Returns a dict: {course_name -> course_dict}
    """
    index = {}
    
    for course in courses:
        name = (course.get('name') or '').strip()
        if not name or name == '':
            continue
        
        # If this course is already in index, keep the one with more skills
        if name in index:
            current_skills = len(index[name].get('skills') or [])
            new_skills = len(course.get('skills') or [])
            if new_skills > current_skills:
                index[name] = course
        else:
            index[name] = course
    
    return index

def build_skill_index(courses: List[Dict]) -> Dict[str, Set[str]]:
    """Build a skill -> set of course names index.
    
    Returns: {skill -> {course_names_that_teach_this_skill}}
    """
    skill_index = defaultdict(set)
    
    for course in courses:
        name = (course.get('name') or '').strip()
        if not name:
            continue
        
        skills = course.get('skills') or []
        for raw_skill in skills:
            if not isinstance(raw_skill, str):
                continue

            # Support two formats:
            # 1) a single skill per list item
            # 2) many skills concatenated in one string separated by double spaces
            if '  ' in raw_skill:
                skill_items = [item.strip() for item in raw_skill.split('  ') if item.strip()]
            else:
                skill_items = [raw_skill.strip()]

            for skill in skill_items:
                skill_normalized = skill.lower()
                if skill_normalized:
                    skill_index[skill_normalized].add(name)
    
    return dict(skill_index)

# =============================================================================
# COURSE SCORING LOGIC (COST, DURATION & DIFFICULTY)
# =============================================================================

def _normalize_token(value: str) -> str:
    """Return a normalized token for robust matching."""
    return (value or '').strip().lower()

def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float with a default fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _course_metrics(course: Dict) -> Dict[str, float]:
    """Extract numeric metrics used for optimization criteria."""
    return {
        'duration_months': _safe_float(course.get('duration_months'), 1.0),
        'cost_usd': _safe_float(course.get('cost_usd'), 0.0),
        'difficulty': _safe_float(course.get('difficulty'), 3.0),
    }

def _score_course_for_criterion(course: Dict, criterion_name: str) -> float:
    """Compute a normalized optimization score for a course under a given criterion.

    Uses min-max domain normalization so all metrics are comparable in [0, 1].
    Lower score = better candidate for that criterion.
    """
    m = _course_metrics(course)

    COST_MAX       = 500.0
    DURATION_MAX   = 6.0
    DIFFICULTY_MAX = 10.0

    cost_n       = max(0.0, min(1.0, m['cost_usd']        / COST_MAX))
    duration_n   = max(0.0, min(1.0, m['duration_months'] / DURATION_MAX))
    difficulty_n = max(0.0, min(1.0, m['difficulty']       / DIFFICULTY_MAX))

    if criterion_name == 'Cheapest path':
        # Minimize total money spent; time and difficulty are minor tiebreakers.
        return (
            0.80 * cost_n
            + 0.12 * duration_n
            + 0.08 * difficulty_n
        )

    if criterion_name == 'Fastest path':
        # Minimize calendar time; harder courses take longer in practice,
        # so difficulty is a meaningful secondary signal. Cost is irrelevant.
        return (
            0.80 * duration_n
            + 0.18 * difficulty_n
            + 0.02 * cost_n
        )

    # 'Balanced path' (default) — equal trade-off, slightly prefer free & shorter.
    return (
        0.45 * duration_n
        + 0.35 * cost_n
        + 0.20 * difficulty_n
    )

# =============================================================================
# UTILITIES: SIMILARITY, SKILL COVERAGE & CATEGORY FILTERS
# =============================================================================


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1, where 1 means identical direction.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm1_sq += a * a
        norm2_sq += b * b

    if norm1_sq == 0.0 or norm2_sq == 0.0:
        return 0.0

    return dot_product / math.sqrt(norm1_sq * norm2_sq)

def _is_skill_covered(skill: str, known_skills: List[str]) -> bool:
    """Return True if a skill is already covered by known skills.

    Uses lexical matching first and falls back to embedding similarity
    when necessary to determine semantic overlap.
    """
    s = _normalize_token(skill)
    if not s:
        return True

    # Fast lexical pass first — no embeddings needed
    known_normalized = [_normalize_token(k) for k in known_skills]
    for k in known_normalized:
        if k and (s == k or s in k or k in s):
            return True

    skill_embedding = get_text_embedding(s)
    if not skill_embedding:
        return False

    for k in known_normalized:
        if not k:
            continue
        known_embedding = get_text_embedding(k)
        if not known_embedding:
            continue
        if _cosine_similarity(skill_embedding, known_embedding) >= SIMILARITY_THRESHOLD:
            return True

    return False

def _passes_category_filter(
    course: Dict,
    avoid_categories: Optional[List[str]],
    embeddings: Dict[str, List[float]]
) -> bool:
    """Reject courses that match avoided categories by name or embedding similarity.

    This preserves course selection quality when users ask to avoid entire topics.
    """

    if not avoid_categories:
        return True

    normalized_avoid_categories = [
        category
        for category in (_normalize_token(item) for item in avoid_categories)
        if category
    ]

    if not normalized_avoid_categories:
        return True

    course_name = (course.get("name") or "").strip()
    if not course_name:
        return True

    course_embedding = embeddings.get(course_name)
    if not course_embedding:
        return True

    for category in normalized_avoid_categories:
        
        if category in _normalize_token(course_name):
            return False
        
        category_embedding = get_text_embedding(category)
        if not category_embedding:
            continue

        similarity = _cosine_similarity(
            course_embedding,
            category_embedding
        )
        
        if similarity >= CATEGORY_SIMILARITY_THRESHOLD:
            return False

    return True


# =============================================================================
# PREREQUISITE & COURSE RESOLVER
# =============================================================================

def find_course_by_skill(
    skill: str,
    skill_index: Dict[str, Set[str]],
    embeddings_data: Optional[Dict[str, List[float]]],
    top_n: int = 6
) -> List[Dict]:
    """
    Hybrid retrieval system that combines semantic embeddings with lexical and index-based signals to rank courses by relevance to a skill query.
    It improves search quality by filtering noise and merging multiple weak signals into a single stable ranking score.

    General flow:
    1. Generates skill embedding and extracts normalized tokens.
    2. Builds candidates using index seeds and semantic similarity over embeddings.
    3. Reranks results by combining lexical overlap, index boosts, and final semantic score.
    """

    def cosine(a, b):
        return _cosine_similarity(a, b)

    def norm(x):
        return max(0.0, min(1.0, (x + 1) / 2))

    def embed(text):
        return get_text_embedding(text)

    def tokenize(text):
        return set(text.lower().split())

    # PREPROCESS
    skill = (skill or "").strip().lower()
    if not skill:
        return []

    skill_tokens = tokenize(skill)
    skill_emb = embed(skill)

    if not skill_emb:
        return []

    # STAGE 1: CANDIDATE SEEDING
    seed = set()

    if skill in skill_index:
        seed.update(skill_index[skill])

    for k, courses in skill_index.items():
        if skill in k or k in skill:
            seed.update(courses)

    # STAGE 2: EMBEDDING RETRIEVAL + STAGE 3: SIGNAL BOOSTING
    candidates = {}

    for course_name, course_emb in (embeddings_data or {}).items():

        if not course_emb:
            continue

        key = course_name.strip().lower()
        course_tokens = tokenize(key)

        sim = cosine(skill_emb, course_emb)
        sim = norm(sim)

        if sim < 0.33:
            continue

        score = sim

        overlap = len(skill_tokens & course_tokens)
        score += overlap * 0.06

        if key in seed:
            score += 0.22

        if skill in key:
            score += 0.18

        if len(course_tokens) < 3:
            score -= 0.05
        if "introduction" in key and overlap == 0:
            score -= 0.07

        score = max(0.0, min(1.0, score))

        candidates[key] = {
            "name": course_name,
            "score": score
        }

    # STAGE 4: GLOBAL RE-RANKING
    results = list(candidates.values())

    for r in results:
        name = r["name"].lower()
        if skill not in name and len(skill_tokens & tokenize(name)) == 0:
            r["score"] *= 0.98

    results.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "name": r["name"],
            "similarity_score": round(r["score"], 4)
        }
        for r in results[:top_n]
    ]

def _get_candidate_courses_for_skill(
    skill: str,
    skill_index: Dict[str, Set[str]],
    embeddings_data: Optional[Dict[str, List[float]]],
    course_index: Dict[str, Dict],
    avoid_categories: Optional[List[str]],
    criterion_name: str,
    top_n: int = 6,
    exclude_courses: Optional[Set[str]] = None,  # <-- nuevo
) -> List[str]:
    """Find and rank candidate courses that can cover a required skill.

    Combines semantic relevance with criterion-based optimization via a
    weighted hybrid score, so the returned list reflects BOTH fit and
    the user's optimization goal (cheapest / fastest / balanced).

    Args:
        exclude_courses: set of course names already present in the current
                         path; filtered out before ranking to prevent duplicates.
    """
    # --- 1. Retrieve semantically relevant candidates ---
    raw_candidates: List[Dict] = []
    try:
        raw_candidates = find_course_by_skill(
            skill=skill,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            top_n=max(12, top_n * 3),
        )
    except Exception:
        raw_candidates = []

    # --- 2. Filter: must exist in index, pass category exclusions, and not be a duplicate ---
    _excluded = exclude_courses or set()
    seen_names: Set[str] = set()   # <-- dedup dentro del propio raw_candidates
    filtered: List[Dict] = []
    for item in raw_candidates:
        name = (item.get('name') or '').strip()
        if not name:
            continue
        if name in seen_names:          # <-- evita duplicados internos del retriever
            continue
        if name not in course_index:
            continue
        if name in _excluded:           # <-- evita cursos ya usados en el path
            continue
        if not _passes_category_filter(
            course_index[name],
            avoid_categories,
            embeddings_data or {},
        ):
            continue
        seen_names.add(name)
        filtered.append(item)

    if not filtered:
        return []

    # --- 3. Compute per-criterion score for each candidate (normalized, lower = better) ---
    criterion_scores: Dict[str, float] = {
        item['name']: _score_course_for_criterion(course_index[item['name']], criterion_name)
        for item in filtered
    }

    min_c = min(criterion_scores.values())
    max_c = max(criterion_scores.values())
    c_range = max_c - min_c if max_c > min_c else 1.0

    def norm_criterion(name: str) -> float:
        return (criterion_scores[name] - min_c) / c_range

    # --- 4. Normalize similarity scores to [0, 1] (higher raw score = more relevant) ---
    sim_scores: Dict[str, float] = {
        item['name']: float(item.get('similarity_score', 0.0))
        for item in filtered
    }
    max_s = max(sim_scores.values()) if sim_scores else 1.0
    min_s = min(sim_scores.values()) if sim_scores else 0.0
    s_range = max_s - min_s if max_s > min_s else 1.0

    def norm_sim(name: str) -> float:
        return (sim_scores[name] - min_s) / s_range

    # --- 5. Hybrid score: relevance pulls up, criterion cost pulls down ---
    CRITERION_WEIGHT = {
        'Cheapest path': 0.38,
        'Fastest path':  0.38,
        'Balanced path': 0.25,
    }
    w_criterion = CRITERION_WEIGHT.get(criterion_name, 0.25)
    w_relevance = 1.0 - w_criterion

    def hybrid_score(name: str) -> float:
        return w_criterion * norm_criterion(name) - w_relevance * norm_sim(name)

    # --- 6. Sort ascending (lower hybrid score = better match + better criterion fit) ---
    ranked = sorted(filtered, key=lambda item: hybrid_score(item['name']))

    return [item['name'] for item in ranked[:top_n]]

def _infer_prereq_skills_for_topic(
    topic: str,
    cache: Dict[str, List[str]],
) -> List[str]:
    """Infer prerequisite skills for a topic using the LLM adapter.

    Caches results by normalized topic to avoid repeated LLM calls.
    """
    normalized_topic = _normalize_token(topic)
    if not normalized_topic:
        return []

    if normalized_topic in cache:
        return cache[normalized_topic]

    try:
        inferred = infer_prerequisites_for_objective(topic) or []
    except Exception:
        inferred = []

    cleaned = []
    seen = set()
    for item in inferred:
        token = _normalize_token(item)
        if not token or token in seen or token == normalized_topic:
            continue
        seen.add(token)
        cleaned.append(item.strip())

    cache[normalized_topic] = cleaned
    return cleaned

def _infer_prereq_skills_for_course(
    course: Dict,
    prereq_cache: Dict[str, List[str]],
) -> List[str]:
    """Infer direct prerequisite skills needed before taking a course.

    Treats beginner-level courses as foundational with no prerequisites.
    """
    level = _normalize_token(course.get('level') or '')
    if 'beginner' in level:
        return []

    name = (course.get('name') or '').strip()
    if not name:
        return []

    prereqs = _infer_prereq_skills_for_topic(name, prereq_cache)
    if prereqs:
        return prereqs

    # Fallback: infer from one representative skill if title-based inference fails.
    skills = [s for s in (course.get('skills') or []) if _normalize_token(s)]
    if skills:
        return _infer_prereq_skills_for_topic(skills[0], prereq_cache)

    return []


# =============================================================================
# PATH GENERATION (RECURSIVE PATH FINDER)
# =============================================================================

def _resolve_route_for_target_course(
    target_course_name: str,
    course_index: Dict[str, Dict],
    skill_index: Dict[str, Set[str]],
    embeddings_data,
    initial_skills: List[str],
    criterion_name: str,
    avoid_categories: Optional[List[str]],
    prereq_cache: Dict[str, List[str]],
    visited: Optional[Set[str]] = None,
) -> Optional[CourseNode]:
    """Recursively resolve prerequisites and build a course dependency tree.

    Returns a CourseNode subtree rooted at the target course, including
    children for inferred prerequisite courses and unresolved skills.
    """
    if visited is None:
        visited = set()

    if target_course_name in visited:
        return None

    course = course_index.get(target_course_name)
    if not course:
        return None

    # Clone visited so sibling branches don't interfere with each other.
    local_visited = set(visited)
    local_visited.add(target_course_name)

    node = CourseNode(name=target_course_name)

    prereq_skills = _infer_prereq_skills_for_course(course, prereq_cache)

    if not prereq_skills:
        return node

    for skill in prereq_skills:

        if _is_skill_covered(skill, initial_skills):
            continue

        # Skills already covered by courses added in this branch
        skills_in_branch: List[str] = []
        for visited_course_name in local_visited:
            visited_course = course_index.get(visited_course_name)
            if visited_course:
                skills_in_branch.extend(visited_course.get('skills') or [])

        if _is_skill_covered(skill, skills_in_branch):
            continue

        candidates = _get_candidate_courses_for_skill(
            skill=skill,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            course_index=course_index,
            avoid_categories=avoid_categories,
            criterion_name=criterion_name,
            top_n=6,
            exclude_courses=local_visited,
        )

        if not candidates:
            node.unresolved.append(skill)
            continue

        node.alternatives.extend(candidates[1:3])

        # Pick the best candidate not already in this branch's visited set.
        best_course = next(
            (c for c in candidates if c not in local_visited),
            None,
        )

        if best_course is None:
            node.unresolved.append(skill)
            continue

        child = _resolve_route_for_target_course(
            target_course_name=best_course,
            course_index=course_index,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            initial_skills=initial_skills,
            criterion_name=criterion_name,
            avoid_categories=avoid_categories,
            prereq_cache=prereq_cache,
            visited=local_visited,
        )

        if child:
            child.skill = skill
            node.children.append(child)
            local_visited.add(best_course)

    return node

def generate_paths(
    courses: List[Dict],
    initial_skills: List[str],
    objective: str,
    max_paths: int = 5,
    avoid_categories: Optional[List[str]] = None,
    user_prefs: Optional[List[str]] = None,
    criterion_name: Optional[str] = None,
    course_index: Optional[Dict[str, Dict]] = None,
    skill_index: Optional[Dict[str, Set[str]]] = None,
    embeddings_data: Optional[Dict[str, List[float]]] = None,
    experiment: bool = False,
    use_strips_planner: bool = True 
) -> List[Dict]:
    """Generate ranked learning paths from initial skills toward an objective.

    Each target course gets its own isolated visited set so the resolver
    can freely pick the best prerequisites per path without being polluted
    by choices made for other paths.

    Final plans are sorted by the active criterion so the best path
    for the user's goal always appears first.
    """
    if not courses and not course_index:
        return []

    initial_skills = initial_skills or []
    user_prefs = user_prefs or []
    criterion = criterion_name if criterion_name else 'Balanced path'

    course_index = course_index or index_courses(courses)
    skill_index = skill_index or build_skill_index(courses)
    embeddings_data = embeddings_data if embeddings_data is not None else _load_embeddings_data(experiment)

    target_courses = _get_candidate_courses_for_skill(
        skill=objective,
        skill_index=skill_index,
        embeddings_data=embeddings_data,
        course_index=course_index,
        avoid_categories=avoid_categories,
        criterion_name=criterion,
        top_n=max_paths*2,
    )

    if not target_courses:
        return []

    plans = []

    for target in target_courses[:max_paths]:

        # Isolated visited set per path — prevents cross-path pollution.
        tree = _resolve_route_for_target_course(
            target_course_name=target,
            course_index=course_index,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            initial_skills=initial_skills,
            criterion_name=criterion,
            avoid_categories=avoid_categories,
            prereq_cache={},
            visited=set(),
        )

        if not tree:
            continue

        ordered_path = CourseNode.flatten(tree)

        # Remove duplicate courses while preserving order. This prevents
        # repeated course entries when the same prerequisite is selected
        # for multiple inferred skills in the same learning path.
        unique_ordered_path = []
        seen_courses = set()
        for name in ordered_path:
            if name not in seen_courses:
                unique_ordered_path.append(name)
                seen_courses.add(name)
        ordered_path = unique_ordered_path

        totals = {'duration': 0.0, 'cost': 0.0, 'difficulty': 0.0}
        for name in ordered_path:
            m = _course_metrics(course_index[name])
            totals['duration']   += m['duration_months']
            totals['cost']       += m['cost_usd']
            totals['difficulty'] += m['difficulty']

        avg_difficulty = totals['difficulty'] / len(ordered_path) if ordered_path else 0.0

        plans.append({
            'target_course': target,
            'course_path':   ordered_path,
            'path':          ordered_path,
            'steps':         ordered_path,
            'user_prefs':    user_prefs,
            'metrics': {
                'total_months':    round(totals['duration'],   2),
                'total_cost':      round(totals['cost'],       2),
                'avg_difficulty':  round(avg_difficulty,       2),
                'steps':           len(ordered_path),
            },
        })

    # ------------------------------------------------------------------ #
    # Re-rank final plans by the active criterion so the best path
    # for the user's goal always surfaces first.
    # ------------------------------------------------------------------ #
    def _plan_sort_key(plan: Dict) -> float:
        m = plan['metrics']
        cost_n       = min(m['total_cost']     / 500.0, 1.0)
        duration_n   = min(m['total_months']   / 6.0,   1.0)
        difficulty_n = min(m['avg_difficulty'] / 10.0,  1.0)

        if criterion == 'Cheapest path':
            return 0.80 * cost_n + 0.12 * duration_n + 0.08 * difficulty_n
        if criterion == 'Fastest path':
            return 0.80 * duration_n + 0.18 * difficulty_n + 0.02 * cost_n
        return 0.45 * duration_n + 0.35 * cost_n + 0.20 * difficulty_n

    plans.sort(key=_plan_sort_key)

    if use_strips_planner and plans:
        plans = refine_paths_with_planner(
            paths=plans,
            initial_skills=initial_skills,
            objective=objective,
            course_index=course_index or index_courses(courses),
            use_strips=True
        )
    
    return plans
