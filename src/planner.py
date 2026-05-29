"""planner.py
Utilities for loading, normalizing and generating learning pathway
plans from a catalog of online courses. Functions include loaders,
normalizers, prerequisite inference, indexing and search heuristics.
"""

import json                        # Loads and dumps JSON data structures
from pathlib import Path           # Object‑oriented filesystem path handling
from typing import List, Dict, Set, Tuple, Optional   # Type hints for common container types
from collections import defaultdict, deque  # Collections for graph data structures


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
    - prerequisites: list of prerequisite skills (inferred or explicit)
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    json_path = data_dir / "normalized_courses.json"
    
    if not json_path.exists():
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            courses = json.load(f)
        if not isinstance(courses, list):
            return []

        # Keep only courses that can actually be referenced in the UI.
        filtered = []
        for course in courses:
            if not isinstance(course, dict):
                continue
            link = str(course.get('link') or course.get('url') or course.get('href') or '').strip()
            name = str(course.get('name') or course.get('nombre') or '').strip()
            if name and link:
                filtered.append(course)
        return filtered
    except Exception as e:
        print(f"Error loading courses from {json_path}: {e}")
        return []

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


# =============================================================================
# PREREQUISITE INFERENCE & GRAPH BUILDING
# =============================================================================

def infer_prerequisites_for_course(
    course: Dict,
    all_courses: List[Dict],
    skill_index: Dict[str, Set[str]]
) -> Set[str]:
    """Infer prerequisite courses for a given course.
    
    A course is a prerequisite if it teaches foundational skills.
    Optimized for performance on large catalogs.
    """
    prerequisites = set()
    
    # Explicit prerequisites field (if exists)
    explicit_prereqs = course.get('prerequisites') or []
    if isinstance(explicit_prereqs, str):
        explicit_prereqs = [p.strip() for p in explicit_prereqs.split(',')]
    for prereq in explicit_prereqs:
        prereq_name = (prereq or '').strip()
        if prereq_name:
            prerequisites.add(prereq_name)
    
    # Quick check: only look for prerequisites for intermediate/advanced courses
    my_level = (course.get('level') or 'Beginner').lower()
    if 'beginner' in my_level:
        return prerequisites
    
    my_difficulty = course.get('difficulty', 5)
    
    # Look for foundational skills (limited set)
    foundational_patterns = [
        'python', 'sql', 'statistics', 'linear algebra',
        'data analysis', 'git'
    ]
    
    for pattern in foundational_patterns:
        if pattern in skill_index and len(prerequisites) < 5:
            # Found courses teaching this foundational skill
            courses_teaching = list(skill_index[pattern])[:10]  # Limit candidates
            for course_name in courses_teaching:
                # Simple heuristic: if name is shorter and not too hard, it's likely foundational
                if len(course_name) < len(course.get('name', 'x')):
                    prerequisites.add(course_name)
    
    return prerequisites

def build_prerequisite_graph(courses: List[Dict]) -> Dict[str, Set[str]]:
    """Build a prerequisite graph: course -> set of prerequisite courses.
    
    Returns: {course_name -> {prerequisite_course_names}}
    
    Optimized: only compute prerequisites for a subset of courses.
    """
    skill_index = build_skill_index(courses)
    prereq_graph = {}
    
    # Only process courses with non-trivial difficulty (avoid 44k full iteration)
    for i, course in enumerate(courses):
        # Sample: process every Nth course + high-difficulty courses
        if i % 50 != 0 and course.get('difficulty', 0) < 7:
            continue
        
        name = (course.get('name') or '').strip()
        if name:
            prereq_graph[name] = infer_prerequisites_for_course(course, [], skill_index)
    
    return prereq_graph


# =============================================================================
# SEARCH HEURISTICS & PATH FINDING
# =============================================================================

def find_courses_by_skill(
    skills: List[str],
    courses: List[Dict],
    skill_index: Dict[str, Set[str]]
) -> Set[str]:
    """Find all courses that teach or relate to the given skills.
    
    Returns a set of course names.
    """
    result = set()
    
    for skill in skills:
        skill_normalized = (skill or '').lower().strip()
        if skill_normalized in skill_index:
            result.update(skill_index[skill_normalized])
        else:
            # Try partial match
            for indexed_skill, courses_teaching in skill_index.items():
                if skill_normalized in indexed_skill or indexed_skill in skill_normalized:
                    result.update(courses_teaching)
    
    return result

def find_courses_by_goal(
    goal: str,
    courses: List[Dict],
    category_index: Dict[str, Set[str]],
    skill_index: Dict[str, Set[str]]
) -> Set[str]:
    """Find courses relevant to a user's goal.
    
    Matches goal against course names, categories, and skills.
    Returns a set of course names.
    """
    result = set()
    goal_lower = (goal or '').lower().strip()
    
    # Match by course name
    for course in courses:
        name = (course.get('name') or '').lower()
        if goal_lower in name or name in goal_lower:
            result.add((course.get('name') or '').strip())
    
    # Match by category
    for category, courses_in_cat in category_index.items():
        if goal_lower in category or category in goal_lower:
            result.update(courses_in_cat)
    
    # Match by skill
    for skill, courses_teaching in skill_index.items():
        if goal_lower in skill or skill in goal_lower:
            result.update(courses_teaching)
    
    return result

def _topological_sort_subset(
    courses_subset: Set[str],
    prereq_graph: Dict[str, Set[str]]
) -> List[str]:
    """Topological sort of courses respecting prerequisite constraints.
    
    Returns a sorted list where prerequisites come before dependents.
    If a prerequisite is not in the subset, it's ignored.
    """
    in_degree = {c: 0 for c in courses_subset}
    adj_list = {c: [] for c in courses_subset}
    
    for course in courses_subset:
        prereqs = prereq_graph.get(course) or set()
        for prereq in prereqs:
            if prereq in courses_subset:
                adj_list[prereq].append(course)
                in_degree[course] += 1
    
    queue = deque([c for c in courses_subset if in_degree[c] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If graph has cycles, return whatever we have
    return result if len(result) == len(courses_subset) else list(courses_subset)

def validate_path_constraints(
    path: List[str],
    courses_index: Dict[str, Dict],
    constraints: Dict
) -> bool:
    """Validate that a path satisfies user constraints.
    
    Constraints dict may include:
    - max_months: maximum total duration
    - max_cost: maximum total cost
    - max_difficulty: maximum average difficulty
    - min_difficulty: minimum average difficulty
    - required_skills: skills that must be covered
    """
    if not path:
        return True
    
    # Calculate metrics
    total_months = sum(courses_index.get(c, {}).get('duration_months', 0) for c in path)
    total_cost = sum(courses_index.get(c, {}).get('cost_usd', 0) for c in path)
    difficulties = [courses_index.get(c, {}).get('difficulty', 5) for c in path]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 5
    
    # Check constraints
    if constraints.get('max_months') and total_months > constraints['max_months']:
        return False
    if constraints.get('max_cost') and total_cost > constraints['max_cost']:
        return False
    if constraints.get('max_difficulty') and avg_difficulty > constraints['max_difficulty']:
        return False
    if constraints.get('min_difficulty') and avg_difficulty < constraints['min_difficulty']:
        return False
    
    # Check required skills coverage
    required_skills = set(
        (s or '').lower().strip() 
        for s in (constraints.get('required_skills') or [])
    )
    if required_skills:
        covered_skills = set()
        for course_name in path:
            course = courses_index.get(course_name, {})
            skills = course.get('skills') or []
            covered_skills.update((s or '').lower().strip() for s in skills)
        
        if not required_skills.issubset(covered_skills):
            return False
    
    return True

# =============================================================================
# PATH GENERATION WITH OPTIMIZATION CRITERIA
# =============================================================================

def _search_path_by_criterion(
    start_courses: Set[str],
    goal_courses: Set[str],
    courses_index: Dict[str, Dict],
    prereq_graph: Dict[str, Set[str]],
    criterion: str = 'balanced',
    constraints: Dict = None,
    max_length: int = 7,
    max_candidates: int = 500
) -> Optional[List[str]]:
    """Search for an optimal path from start_courses towards goal_courses.
    
    Criterion: 'fastest', 'cheapest', 'easiest', 'balanced'
    Uses greedy search with limited neighborhood.
    Optimized for large course catalogs.
    """
    if not constraints:
        constraints = {}
    
    # Limit goal courses to top candidates
    goal_list = list(goal_courses)[:max_candidates]
    if not goal_list:
        return None
    
    # Start with the best goal course
    best_course = min(
        goal_list,
        key=lambda c: _score_course(c, courses_index, criterion)
    )
    
    path = [best_course]
    visited = {best_course}
    
    # Greedy expansion: add courses that improve the path
    for iteration in range(max_length - 1):
        current_course = path[-1]
        
        # Get limited set of candidates
        candidates = set()
        
        # Add prerequisites (if any)
        prereqs = prereq_graph.get(current_course, set())
        if prereqs:
            candidates.update(list(prereqs)[:5])
        
        # Add related courses by skill
        skills = courses_index.get(current_course, {}).get('skills', [])
        if skills:
            # Try multiple skills to get more diversity
            for skill in skills[:5]:
                skill_lower = (skill or '').lower().strip()
                # Look for other courses with similar skills
                for cname in list(courses_index.keys())[::max(1, len(courses_index) // 500)]:
                    if cname not in visited:
                        cdata = courses_index.get(cname, {})
                        cskills = cdata.get('skills', [])
                        if any((s or '').lower().strip() == skill_lower for s in cskills):
                            candidates.add(cname)
                            if len(candidates) >= 50:
                                break
                if len(candidates) >= 50:
                    break
        
        # Also add courses from same category (for diversity)
        current_category = courses_index.get(current_course, {}).get('category', '').lower()
        if current_category:
            for cname, cdata in list(courses_index.items())[::max(1, len(courses_index) // 300)]:
                if cname not in visited and cdata.get('category', '').lower() == current_category:
                    candidates.add(cname)
                    if len(candidates) >= 50:
                        break
        
        # Filter out courses that violate constraints
        candidates = list(candidates - visited)
        valid_candidates = []
        for cand in candidates:
            if validate_path_constraints(path + [cand], courses_index, constraints):
                valid_candidates.append(cand)
        
        candidates = valid_candidates[:30]  # Keep top 30
        
        if not candidates:
            break
        
        # Select best candidate
        best_next = min(
            candidates,
            key=lambda c: _score_course(c, courses_index, criterion)
        )
        
        path.append(best_next)
        visited.add(best_next)
    
    # Ensure minimum path length for meaningful trayectories
    return path if len(path) >= 2 else None

def _score_course(course_name: str, courses_index: Dict[str, Dict], criterion: str) -> float:
    """Score a single course based on criterion."""
    course = courses_index.get(course_name, {})
    
    if criterion == 'fastest':
        return course.get('duration_months', 6)
    elif criterion == 'cheapest':
        return course.get('cost_usd', 100)
    elif criterion == 'easiest':
        return -course.get('difficulty', 5)  # Negative for min-heap
    elif criterion == 'balanced':
        # Weighted score
        months = course.get('duration_months', 6) / 12  # Normalize to years
        cost = course.get('cost_usd', 100) / 1000
        difficulty = course.get('difficulty', 5) / 10
        return (months + cost + difficulty) / 3
    else:
        return course.get('difficulty', 5)

def _score_path(path: List[str], courses_index: Dict[str, Dict], criterion: str) -> float:
    """Score an entire path based on criterion."""
    total_months = sum(courses_index.get(c, {}).get('duration_months', 0) for c in path)
    total_cost = sum(courses_index.get(c, {}).get('cost_usd', 0) for c in path)
    difficulties = [courses_index.get(c, {}).get('difficulty', 5) for c in path]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 5
    
    if criterion == 'fastest':
        return total_months
    elif criterion == 'cheapest':
        return total_cost
    elif criterion == 'easiest':
        return -avg_difficulty  # Negative for min-heap
    elif criterion == 'balanced':
        months_norm = total_months / (len(path) * 12) if path else 1
        cost_norm = total_cost / (len(path) * 500) if path else 1
        diff_norm = avg_difficulty / 10
        return (months_norm + cost_norm + diff_norm) / 3
    else:
        return len(path)

def _calculate_path_metrics(path: List[str], courses_index: Dict[str, Dict]) -> Dict:
    """Calculate metrics for a path."""
    metrics = {
        'total_months': 0,
        'total_cost': 0,
        'avg_difficulty': 0,
        'steps': len(path)
    }
    
    if not path:
        return metrics
    
    total_months = 0
    total_cost = 0
    difficulties = []
    
    for course_name in path:
        course = courses_index.get(course_name, {})
        total_months += course.get('duration_months', 0)
        total_cost += course.get('cost_usd', 0)
        difficulties.append(course.get('difficulty', 5))
    
    metrics['total_months'] = total_months
    metrics['total_cost'] = total_cost
    metrics['avg_difficulty'] = sum(difficulties) / len(difficulties) if difficulties else 0
    
    return metrics

def generate_paths(
    courses_list: List[Dict] = None,
    skills: List[str] = None,
    goal: str = None,
    max_paths: int = 5,
    avoid_categories: Set[str] = None,
    user_prefs: Dict = None,
    criteria_names: List[str] = None,
    num_paths: int = None,
    preferences: List[str] = None,
    constraints: Dict = None
) -> List[Dict]:
    """Generate multiple alternative career/learning pathways.
    
    Compatible with both direct calls and app.py interface.
    Main entry point for path generation. Explores different optimization
    criteria and returns a ranked list of pathways.
    
    Args:
        courses_list: Optional pre-loaded list of courses (for compatibility)
        skills: Existing skills of the user
        goal: Target career/role (e.g., "Data Scientist", "ML Engineer")
        max_paths: Maximum number of paths to generate
        avoid_categories: Set of categories to avoid
        user_prefs: User preferences dict
        criteria_names: List of criterion names to use
        num_paths: Alternative parameter for max_paths (legacy)
        preferences: User preferences list
        constraints: Dict with max_months, max_cost, max_difficulty, etc.
    
    Returns:
        List of dicts with keys:
        - criterion: "Fastest path", "Cheapest path", etc.
        - path: List of course names (for UI)
        - course_path: List of course objects
        - steps: List of (milestone, course) tuples
        - metrics: Dict with timing, cost, difficulty stats
        - target_course: Target course (for UI compatibility)
    """
    # Handle parameter compatibility
    if num_paths is not None and max_paths == 5:
        max_paths = num_paths
    
    if not constraints:
        constraints = {}
    if not skills:
        skills = []
    if not preferences:
        preferences = []
    if not avoid_categories:
        avoid_categories = set()
    if not user_prefs:
        user_prefs = {}
    if not criteria_names:
        criteria_names = ['rapida', 'economica', 'balanceada']
    
    # Load data
    if courses_list is None:
        courses = load_courses()
    else:
        courses = courses_list
    
    if not courses:
        return []
    
    courses_index = index_courses(courses)
    skill_index = build_skill_index(courses)
    category_index = build_category_index(courses)
    prereq_graph = build_prerequisite_graph(courses)
    
    # Filter out courses from avoided categories
    if avoid_categories:
        courses_index = {
            name: data 
            for name, data in courses_index.items()
            if (data.get('category', '').lower() not in {c.lower() for c in avoid_categories})
        }
    
    # Find relevant courses (limited to avoid combinatorial explosion)
    goal_courses = find_courses_by_goal(goal or '', courses, category_index, skill_index)
    skill_courses = find_courses_by_skill(skills or [], courses, skill_index)
    
    # Limit goal courses to top candidates
    if goal_courses:
        # Rank by relevance (prefer courses with more skills + lower difficulty)
        goal_list = sorted(
            goal_courses,
            key=lambda c: (
                -len(courses_index.get(c, {}).get('skills', [])),
                courses_index.get(c, {}).get('difficulty', 10)
            )
        )[:100]  # Limit to 100 goal courses
        goal_courses = set(goal_list)
    else:
        goal_courses = skill_courses if skill_courses else set()
    
    if not goal_courses:
        # Fallback: pick diverse courses from categories
        for cat in list(category_index.keys())[:3]:
            goal_courses.update(list(category_index[cat])[:10])
        if not goal_courses:
            goal_courses = set(list(courses_index.keys())[:20])
    
    # Map criteria names to optimization criteria
    criteria_mapping = {
        'rapida': 'fastest',
        'economica': 'cheapest',
        'balanceada': 'balanced',
        'facil': 'easiest',
        'premium': 'balanced'
    }
    
    criteria_to_use = []
    for name in (criteria_names or ['rapida', 'economica', 'balanceada']):
        criterion = criteria_mapping.get(name.lower(), name.lower())
        if criterion not in criteria_to_use:
            criteria_to_use.append(criterion)
    
    # Ensure we have some criteria
    if not criteria_to_use:
        criteria_to_use = ['fastest', 'cheapest', 'balanced']
    
    # Generate paths under different criteria
    generated_paths = []
    seen_paths = set()
    
    for criterion in criteria_to_use:
        path = _search_path_by_criterion(
            start_courses=skill_courses,
            goal_courses=goal_courses,
            courses_index=courses_index,
            prereq_graph=prereq_graph,
            criterion=criterion,
            constraints=constraints,
            max_length=7,
            max_candidates=200
        )
        
        if path:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                metrics = _calculate_path_metrics(path, courses_index)
                
                # Create milestone steps
                steps = []
                for i, course_name in enumerate(path):
                    course_data = courses_index.get(course_name, {})
                    if i == 0:
                        steps.append((f"Foundation: {course_name}", course_data))
                    elif i == len(path) - 1:
                        steps.append((f"Capstone: {course_name}", course_data))
                    elif i == len(path) // 2:
                        steps.append((f"Core: {course_name}", course_data))
                    else:
                        steps.append((f"Build: {course_name}", course_data))
                
                criterion_label = {
                    'fastest': 'Fastest path',
                    'cheapest': 'Cheapest path',
                    'easiest': 'Easiest path',
                    'balanced': 'Balanced path'
                }.get(criterion, f'Path ({criterion})')
                
                generated_paths.append({
                    'criterion': criterion_label,
                    'path': path,
                    'course_path': [courses_index.get(c, {}) for c in path],
                    'steps': steps,
                    'metrics': metrics,
                    'target_course': goal or 'Career Path'
                })
    
    # Sort by total time for better UX
    generated_paths.sort(key=lambda p: p.get('metrics', {}).get('total_months', float('inf')))
    
    return generated_paths[:max_paths]
