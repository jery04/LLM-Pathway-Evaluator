"""
STRIPS planner with BFS and A* search for learning path optimization.

What this file does:
- Defines planning problems (facts, operators, initial state, goals)
- Checks if an operator can be applied to a state
- Applies operators to get new states
- Tests if a state satisfies the goal
- Searches for a plan using BFS or A*
- Bridges course data to STRIPS planning

Main functions:
- solve_bfs() → finds a plan using Breadth-First Search
- solve_astar() → finds a plan using A* (usually faster)
- optimize_path_with_strips() → main bridge function for planner.py
"""

from dataclasses import dataclass          # Creates simple data container classes
from typing import Set, List, Optional, Dict  # Type hints for function signatures
from collections import deque              # Efficient double-ended queue for BFS
from heapq import heappush, heappop        # Priority queue operations for A* search


# ==========================================================
# STRIPS PLANNING ENTITIES
# ==========================================================

@dataclass
class Operator:
    """STRIPS operator with preconditions and effects."""
    
    name: str
    positive_preconditions: Set[str]
    add_effects: Set[str]
    cost: float = 1.0

@dataclass
class STRIPSProblem:
    """STRIPS planning problem <P,O,I,G>."""
    
    propositions: Set[str]
    operators: List[Operator]
    initial_state: Set[str]
    positive_goals: Set[str]

@dataclass
class Plan:
    """Represents a solution plan."""
    
    actions: List[str]
    final_state: Set[str]
    total_cost: float


# ==========================================================
# STRIPS STATE TRANSITION FUNCTIONS
# ==========================================================

def applicable(state: Set[str], operator: Operator) -> bool:
    """Check whether an operator can be applied to a state."""
    return operator.positive_preconditions.issubset(state)

def successor(state: Set[str], operator: Operator) -> Optional[Set[str]]:
    """Apply an operator and return the successor state."""
    if not applicable(state, operator):
        return None
    
    new_state = set(state)
    new_state |= operator.add_effects
    return new_state

def goal_test(state: Set[str], positive_goals: Set[str]) -> bool:
    """Check whether a state satisfies the goal."""
    return positive_goals.issubset(state)

def heuristic(state: Set[str], goal_facts: Set[str]) -> int:
    """Estimate remaining distance to the goal (number of missing facts)."""
    return len(goal_facts - state)


# ==========================================================
# STRIPS SOLUTION SEARCH (BFS AND A*)
# ==========================================================

def solve_bfs(problem: STRIPSProblem, max_depth: int = 20) -> Optional[Plan]:
    """Solve the planning problem using BFS."""
    
    start_state = frozenset(problem.initial_state)
    queue = deque([(start_state, [], 0.0)])
    visited = {start_state}
    
    while queue:
        state, actions, cost = queue.popleft()
        current_state = set(state)
        
        if goal_test(current_state, problem.positive_goals):
            return Plan(actions=actions, final_state=current_state, total_cost=cost)
        
        if len(actions) >= max_depth:
            continue
        
        for op in problem.operators:
            next_state = successor(current_state, op)
            if next_state is None:
                continue
            
            frozen = frozenset(next_state)
            if frozen in visited:
                continue
            
            visited.add(frozen)
            queue.append((frozen, actions + [op.name], cost + op.cost))
    
    return None

def solve_astar(problem: STRIPSProblem, max_depth: int = 20) -> Optional[Plan]:
    """Solve the planning problem using A* search."""
    
    start_state = frozenset(problem.initial_state)
    start_h = heuristic(set(start_state), problem.positive_goals)
    
    frontier = [(start_h, 0.0, start_state, [])]
    best_cost = {start_state: 0.0}
    
    while frontier:
        f, g, state, actions = heappop(frontier)
        current_state = set(state)
        
        if goal_test(current_state, problem.positive_goals):
            return Plan(actions=actions, final_state=current_state, total_cost=g)
        
        if len(actions) >= max_depth:
            continue
        
        for op in problem.operators:
            next_state = successor(current_state, op)
            if next_state is None:
                continue
            
            frozen = frozenset(next_state)
            new_g = g + op.cost
            
            if frozen in best_cost and best_cost[frozen] <= new_g:
                continue
            
            best_cost[frozen] = new_g
            h = heuristic(next_state, problem.positive_goals)
            heappush(frontier, (new_g + h, new_g, frozen, actions + [op.name]))
    
    return None


# ==========================================================
# BRIDGE FUNCTIONS FOR PLANNER INTEGRATION
# ==========================================================

def create_operator_from_course(
    course_name: str,
    course_skills: List[str],
    course_prereqs: List[str],
    cost: float = 1.0
) -> Operator:
    """Convert a course into a STRIPS operator.
    
    - Prerequisites become positive preconditions
    - Skills taught become add effects
    """
    return Operator(
        name=f"take_{course_name.replace(' ', '_')}",
        positive_preconditions=set(course_prereqs),
        add_effects=set(course_skills),
        cost=cost
    )

def create_problem_from_paths(
    initial_skills: List[str],
    target_goal: str,
    courses_data: List[Dict],
    prerequisite_map: Dict[str, List[str]]
) -> STRIPSProblem:
    """Create a STRIPS planning problem from course data.
    
    Args:
        initial_skills: Skills the user already has
        target_goal: The learning objective (treated as a skill to achieve)
        courses_data: List of course dicts with name and skills
        prerequisite_map: Mapping course_name -> prerequisite_skills
    
    Returns:
        STRIPSProblem ready for planning
    """
    propositions = set(initial_skills)
    operators = []
    
    for course in courses_data:
        name = course.get('name', '').strip()
        if not name:
            continue
        
        skills = course.get('skills', [])
        prereqs = prerequisite_map.get(name, [])
        
        propositions.update(skills)
        propositions.update(prereqs)
        
        operators.append(create_operator_from_course(
            course_name=name,
            course_skills=skills,
            course_prereqs=prereqs,
            cost=course.get('duration_months', 1.0)
        ))
    
    propositions.add(target_goal)
    
    return STRIPSProblem(
        propositions=propositions,
        operators=operators,
        initial_state=set(initial_skills),
        positive_goals={target_goal}
    )

def plan_to_course_sequence(plan: Plan, course_names: List[str]) -> List[str]:
    """Convert a STRIPS plan back to a sequence of course names.
    
    Assumes operator names follow the pattern "take_{course_name}"
    """
    course_sequence = []
    
    for action in plan.actions:
        if action.startswith("take_"):
            course_name = action[5:].replace('_', ' ')
            if course_name in course_names:
                course_sequence.append(course_name)
    
    return course_sequence

def optimize_path_with_strips(
    initial_skills: List[str],
    goal: str,
    candidate_paths: List[Dict],
    course_index: Dict[str, Dict],
    prerequisite_cache: Optional[Dict[str, List[str]]] = None
) -> List[Dict]:
    """Use STRIPS planning to refine and optimize candidate paths.
    
    This function takes the paths generated by generate_paths and
    tries to find a more optimal sequence using BFS or A*.
    """
    if not candidate_paths:
        return []
    
    prereq_map = prerequisite_cache or {}
    
    # Collect all courses involved in candidate paths
    relevant_courses = set()
    for path in candidate_paths:
        for course_name in path.get('path', []):
            relevant_courses.add(course_name)
    
    # Build course data list for relevant courses
    courses_data = []
    for course_name in relevant_courses:
        if course_name in course_index:
            courses_data.append(course_index[course_name])
    
    if not courses_data:
        return candidate_paths
    
    # Create STRIPS problem
    problem = create_problem_from_paths(
        initial_skills=initial_skills,
        target_goal=goal,
        courses_data=courses_data,
        prerequisite_map=prereq_map
    )
    
    # Try A* first (usually faster), fallback to BFS
    plan = solve_astar(problem, max_depth=20)
    if plan is None:
        plan = solve_bfs(problem, max_depth=20)
    
    if plan is None:
        return candidate_paths
    
    # Convert plan back to course sequence
    optimized_sequence = plan_to_course_sequence(plan, list(relevant_courses))
    if not optimized_sequence:
        return candidate_paths
    
    # Create optimized path structure
    optimized_path = {
        'target_course': goal,
        'course_path': optimized_sequence,
        'path': optimized_sequence,
        'steps': optimized_sequence,
        'user_prefs': candidate_paths[0].get('user_prefs', []) if candidate_paths else [],
        'metrics': {
            'total_months': plan.total_cost,
            'total_cost': 0,
            'avg_difficulty': 0,
            'steps': len(optimized_sequence),
        },
        'planning_method': 'STRIPS_A*' if plan else 'STRIPS_BFS'
    }
    
    # Recalculate metrics for the optimized path
    total_cost = 0.0
    total_difficulty = 0.0
    for course_name in optimized_sequence:
        if course_name in course_index:
            course = course_index[course_name]
            total_cost += course.get('cost_usd', 0)
            total_difficulty += course.get('difficulty', 5)
    
    optimized_path['metrics']['total_cost'] = round(total_cost, 2)
    if optimized_sequence:
        optimized_path['metrics']['avg_difficulty'] = round(total_difficulty / len(optimized_sequence), 2)
    
    # Insert optimized path at the beginning, keep original as alternatives
    return [optimized_path] + candidate_paths[:3]

def refine_paths_with_planner(
    paths: List[Dict],
    initial_skills: List[str],
    objective: str,
    course_index: Dict[str, Dict],
    use_strips: bool = True
) -> List[Dict]:
    """Apply STRIPS planning as a refinement layer on top of generated paths.

    Takes candidate paths and finds a more optimal sequence using classical AI planning (A*/BFS).
    Returns refined paths with STRIPS-optimized plan as first element, or original paths if planning fails.
    """
    if not use_strips or not paths:
        return paths
    
    try:
        # Build prerequisite cache from existing paths
        prereq_cache = {}
        
        # Use STRIPS to optimize the best path
        optimized = optimize_path_with_strips(
            initial_skills=initial_skills,
            goal=objective,
            candidate_paths=paths,
            course_index=course_index,
            prerequisite_cache=prereq_cache
        )
        
        return optimized if optimized else paths
        
    except Exception as e:
        # If STRIPS fails, gracefully fall back to original paths
        print(f"Warning: STRIPS refinement failed: {e}")
        return paths
    
