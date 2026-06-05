"""planner.py
Utilities for loading, normalizing and generating learning pathway
plans from a catalog of online courses. Functions include loaders,
normalizers, prerequisite inference, indexing and search heuristics.
"""

import json                        # Loads and dumps JSON data structures
from pathlib import Path           # Object‑oriented filesystem path handling
from typing import List, Dict, Set, Optional   # Type hints for common container types
from collections import defaultdict, deque  # Collections for graph data structures
import math                        # Math functions for cosine similarity
from dataclasses import dataclass, field

# Removed accidental/unused imports that cause runtime errors when
# optional deps (sympy/torch) are not installed.


# Import LLM adapter for prerequisite inference and embeddings
try:
    from llm_adapter import infer_prerequisites_for_objective, get_text_embedding
except ImportError:
    # Fallback if import fails
    infer_prerequisites_for_objective = None
    get_text_embedding = None

SIMILARITY_THRESHOLD = 0.75
CATEGORY_SIMILARITY_THRESHOLD = 0.6

# =============================================================================
# DATA LOADING & NORMALIZATION
# =============================================================================

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

def _load_embeddings_data() -> Optional[Dict[str, List[float]]]:
    """Load embeddings from data/embedding.json as {name: embedding}."""
    embedding_path = (
        Path(__file__).resolve().parent.parent
        / 'data'
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
        for skill in skills:
            skill_normalized = (skill or '').strip().lower()
            if skill_normalized:
                skill_index[skill_normalized].add(name)
    
    return dict(skill_index)

def build_category_index(courses: List[Dict]) -> Dict[str, Set[str]]:
    """Build a category -> set of course names index.
    
    Returns: {category -> {course_names_in_this_category}}
    """
    category_index = defaultdict(set)
    
    for course in courses:
        name = (course.get('name') or '').strip()
        if not name:
            continue
        
        category = (course.get('category') or 'Uncategorized').strip().lower()
        category_index[category].add(name)
    
    return dict(category_index)

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Returns a value between -1 and 1, where 1 means identical direction.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def find_course_by_skill(
    skill: str,
    skill_index: Dict[str, Set[str]],
    embeddings_data: Optional[Dict[str, List[float]]],
    top_n: int = 3
) -> List[Dict]:
    """Find courses most similar to a given skill using embeddings.
    
    First checks the provided skill index and, if the skill exists there,
    adds all matching courses ordered by course-name similarity to the skill.
    Then it continues with the embedding-based search over embedding.json.
    
    Args:
        skill: The skill or topic to search for
        courses: Preloaded course list
        skill_index: Precomputed skill -> courses index
        embeddings_data: Preloaded embeddings map: {course_name -> embedding}
        top_n: Number of top courses to return (default: 3)
    
    Returns:
        List of course dictionaries with keys: name, similarity_score
    """
    
    skill_normalized = (skill or '').strip().lower()
    similarities = []
    seen_courses = set()

    if skill_normalized and skill_index.get(skill_normalized):
        for course_name in skill_index[skill_normalized]:
            course_name_normalized = (course_name or '').strip().lower()
            if not course_name_normalized or course_name_normalized in seen_courses:
                continue

            similarity = _cosine_similarity(get_text_embedding(course_name.strip().lower()),get_text_embedding(skill_normalized))
            similarities.append({
                'name': course_name,
                'similarity_score': similarity
            })
            seen_courses.add(course_name_normalized)

    try:

        if not embeddings_data:
            return similarities
        
        # Generate embedding for the skill
        skill_embedding = get_text_embedding(skill)
        if not skill_embedding:
            return similarities
        
        # Calculate similarity with all courses
        for course_name, course_embedding in embeddings_data.items():
            course_name_normalized = (course_name or '').strip().lower()
            
            if not course_name_normalized or course_name_normalized in seen_courses or not course_embedding:
                continue
            
            # Calculate cosine similarity
            similarity = _cosine_similarity(skill_embedding, course_embedding)
            
            similarities.append({
                'name': course_name,
                'similarity_score': similarity
            })
            seen_courses.add(course_name_normalized)
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_n]
    
    except Exception as e:
        print(f"Error finding courses by skill '{skill}': {e}")
        return similarities


# =============================================================================
# PATH PLANNING LOGIC (DAG WITH RECURSIVE PREREQUISITES AND ALTERNATIVES)
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
    """Compute a scalar optimization score for a single course."""
    m = _course_metrics(course)
    # Criteria are handled strictly in English and compared by their original labels.
    if criterion_name == 'Cheapest path':
        return m['cost_usd'] + 0.35 * m['duration_months'] + 0.2 * m['difficulty']
    if criterion_name == 'Fastest path':
        return m['duration_months'] + 0.001 * m['cost_usd'] + 0.25 * m['difficulty']

    # Balanced: simple weighted combination for robust ranking.
    return 0.5 * m['duration_months'] + 0.002 * m['cost_usd'] + 0.4 * m['difficulty']

def _sort_courses_by_criterion(
    course_names: List[str],
    course_index: Dict[str, Dict],
    criterion_name: str,
) -> List[str]:
    """Return course names ordered from best to worst for criterion."""
    return sorted(
        [name for name in course_names if name in course_index],
        key=lambda name: _score_course_for_criterion(course_index[name], criterion_name),
    )

def _is_skill_covered(skill: str, known_skills: List[str]) -> bool:
    """Return True if a skill is already covered by user-known skills."""

    s = _normalize_token(skill)
    if not s:
        return True

    skill_embedding = get_text_embedding(s)
    if not skill_embedding:
        return False

    for known in known_skills:
        k = _normalize_token(known)
        if not k:
            continue

        # Fast lexical match
        if s == k or s in k or k in s:
            return True

        known_embedding = get_text_embedding(k)
        if not known_embedding:
            continue

        similarity = _cosine_similarity(
            skill_embedding,
            known_embedding
        )

        if similarity >= SIMILARITY_THRESHOLD:
            return True

    return False

def _passes_category_filter(
    course: Dict,
    avoid_categories: Optional[List[str]],
    embeddings: Dict[str, List[float]]
) -> bool:
    """Reject courses whose embeddings are too close to avoided categories."""

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

def _get_candidate_courses_for_skill(
    skill: str,
    skill_index: Dict[str, Set[str]],
    embeddings_data: Optional[Dict[str, List[float]]],
    course_index: Dict[str, Dict],
    avoid_categories: Optional[List[str]],
    criterion_name: str,
    top_n: int = 4,
) -> List[str]:
    """Find and rank candidate courses that can cover a required skill."""
    candidates = []
    try:
        candidates = find_course_by_skill(
            skill=skill,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            top_n=max(6, top_n),
        )
    except Exception:
        candidates = []

    names = []
    for item in candidates:
        name = (item.get('name') or '').strip()
        if name not in course_index:
            continue
        if not _passes_category_filter(
            course_index[name],
            avoid_categories,
            embeddings_data or {},
        ):
            continue
        names.append(name)

    names = _sort_courses_by_criterion(names, course_index, criterion_name)
    return names[:top_n]

def _infer_prereq_skills_for_topic(
    topic: str,
    cache: Dict[str, List[str]],
) -> List[str]:
    """Infer prerequisite skills for a topic using existing LLM adapter function."""
    normalized_topic = _normalize_token(topic)
    if not normalized_topic:
        return []

    if normalized_topic in cache:
        return cache[normalized_topic]

    if infer_prerequisites_for_objective is None:
        cache[normalized_topic] = []
        return []

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
    """Infer direct prerequisite skills for a course.

    Beginner courses are considered foundational and return no dependencies.
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


# =========================
# 1. ESTRUCTURA DEL ÁRBOL
# =========================

@dataclass
class CourseNode:
    name: str
    skill: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    children: List["CourseNode"] = field(default_factory=list)

    @staticmethod
    def flatten(node: "CourseNode") -> List[str]:
        result = []
        for child in node.children:
            result.extend(CourseNode.flatten(child))
        result.append(node.name)
        return result
    
    @staticmethod
    def print_tree(node: "CourseNode", indent: int = 0) -> None:
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

    if visited is None:
        visited = set()

    if target_course_name in visited:
        return None

    course = course_index.get(target_course_name)
    if not course:
        return None

    visited.add(target_course_name)

    node = CourseNode(name=target_course_name)

    prereq_skills = _infer_prereq_skills_for_course(course, prereq_cache)

    if not prereq_skills:
        return node

    for skill in prereq_skills:

        # si el usuario ya domina la skill
        if _is_skill_covered(skill, initial_skills):
            continue

        candidates = _get_candidate_courses_for_skill(
            skill=skill,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            course_index=course_index,
            avoid_categories=avoid_categories,
            criterion_name=criterion_name,
            top_n=3,
        )

        if not candidates:
            node.unresolved.append(skill)
            continue

        # guardamos alternativas (no usadas)
        node.alternatives.extend(candidates[1:3])

        best_course = candidates[0]

        child = _resolve_route_for_target_course(
            target_course_name=best_course,
            course_index=course_index,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            initial_skills=initial_skills,
            criterion_name=criterion_name,
            avoid_categories=avoid_categories,
            prereq_cache=prereq_cache,
            visited=visited,
        )

        if child:
            child.skill = skill
            node.children.append(child)

    return node

def generate_paths(
    courses: List[Dict],
    initial_skills: List[str],
    objective: str,
    max_paths: int = 5,
    avoid_categories: Optional[List[str]] = None,
    user_prefs: Optional[List[str]] = None,
    criterion_name: Optional[str] = None,
) -> List[Dict]:

    if not courses:
        return []

    initial_skills = initial_skills or []
    user_prefs = user_prefs or []

    criterion = criterion_name if criterion_name else 'Balanced path'
    
    course_index = index_courses(courses)
    skill_index = build_skill_index(courses)
    embeddings_data = _load_embeddings_data()

    target_courses = _get_candidate_courses_for_skill(
        skill=objective,
        skill_index=skill_index,
        embeddings_data=embeddings_data,
        course_index=course_index,
        avoid_categories=avoid_categories,
        criterion_name=criterion,
        top_n=max_paths,
    )

    if not target_courses:
        return []

    plans = []
    visited_global = set()

    for target in target_courses[:max_paths]:

        tree = _resolve_route_for_target_course(
            target_course_name=target,
            course_index=course_index,
            skill_index=skill_index,
            embeddings_data=embeddings_data,
            initial_skills=initial_skills,
            criterion_name=criterion,
            avoid_categories=avoid_categories,
            prereq_cache={},
            visited=visited_global,
        )

        if not tree:
            continue

        ordered_path = CourseNode.flatten(tree)

        totals = {
            "duration": 0.0,
            "cost": 0.0,
            "difficulty": 0.0
        }

        for name in ordered_path:
            metrics = _course_metrics(course_index[name])
            totals["duration"] += metrics["duration_months"]
            totals["cost"] += metrics["cost_usd"]
            totals["difficulty"] += metrics["difficulty"]

        avg_difficulty = (
            totals["difficulty"] / len(ordered_path)
            if ordered_path
            else 0.0
        )

        plans.append({
            "target_course": target,
            "course_path": ordered_path,
            "path": ordered_path,
            "steps": ordered_path,
            "user_prefs": user_prefs,
            "metrics": {
                "total_months": round(totals["duration"], 2),
                "total_cost": round(totals["cost"], 2),
                "avg_difficulty": round(avg_difficulty, 2),
                "steps": len(ordered_path),
            },
        })

        if len(plans) >= max_paths:
            break

    return plans



# TEST
def main():
    # 1. Cargar todos los datos necesarios
    courses = load_courses()
    course_index = index_courses(courses)
    skill_index = build_skill_index(courses)
    embeddings_data = _load_embeddings_data()


    # 2. Definir skills iniciales del usuario
    initial_skills = [
        "Python programming",
        "Basic mathematics",
    ]
    
    # 3. Seleccionar un curso objetivo (primer curso disponible como ejemplo)
    target_course_name = "Machine Learning"
    
    # 4. Definir criterios de optimización
    criterion_name = "Balanced path"  # Opciones: "Cheapest path", "Fastest path", "Balanced path"
    
    # 5. Categorías a evitar (opcional)
    avoid_categories = []
    
    # 6. Cache para prerequisitos (inicialmente vacío)
    prereq_cache = {}
    
    
    target_courses = _get_candidate_courses_for_skill(
        skill=target_course_name,
        skill_index=skill_index,
        embeddings_data=embeddings_data,
        course_index=course_index,
        avoid_categories=avoid_categories,
        criterion_name=criterion_name,
        top_n=5,
    )
    
    # 7. Ejecutar el test con todos los parámetros
    tree = _resolve_route_for_target_course(
        target_course_name=target_courses[0],
        course_index=course_index,
        skill_index=skill_index,
        embeddings_data=embeddings_data,
        initial_skills=initial_skills,
        criterion_name=criterion_name,
        avoid_categories=avoid_categories,
        prereq_cache=prereq_cache,
        visited=None  # El método crea uno si es None
    )
    
    # 8. Imprimir el árbol de rutas
    if tree:
        print(f"\n✅ Árbol de ruta generado para: {target_course_name}\n")
        CourseNode.print_tree(tree)
    else:
        print(f"\n❌ No se pudo generar ruta para: {target_course_name}")
               
