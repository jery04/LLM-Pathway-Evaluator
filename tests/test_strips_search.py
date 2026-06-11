"""
Test suite for STRIPS planning components (simplified version).
Tests core functionality: operators, state transitions, BFS, and A* search.
"""

import sys
import os
from collections import deque
from heapq import heappush, heappop
from typing import Set, List, Optional
from dataclasses import dataclass
import unittest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))  # Add src to path 


from strips_search import   (
    Operator, STRIPSProblem,
    applicable,
    successor,
    goal_test,
    heuristic,
    solve_bfs,
    solve_astar
)


# ==========================================================
# TEST SUITE 
# ==========================================================

class TestSTRIPSBasics(unittest.TestCase):
    """Basic tests for STRIPS components"""
    
    def test_applicable_operator(self):
        """Operator is applicable when all preconditions are met"""
        state = {"python", "git"}
        op = Operator("test", positive_preconditions={"python"}, add_effects=set())
        
        self.assertTrue(applicable(state, op))
        
        # Not applicable when missing precondition
        op2 = Operator("test2", positive_preconditions={"java"}, add_effects=set())
        self.assertFalse(applicable(state, op2))
    
    def test_successor_state(self):
        """Successor adds effects to state"""
        state = {"a", "b"}
        op = Operator("test", positive_preconditions={"a"}, add_effects={"c", "d"})
        
        new_state = successor(state, op)
        
        self.assertEqual(new_state, {"a", "b", "c", "d"})
    
    def test_goal_test(self):
        """Goal is achieved when all goals are in state"""
        state = {"python", "java", "javascript"}
        
        self.assertTrue(goal_test(state, {"python", "javascript"}))
        self.assertFalse(goal_test(state, {"python", "ruby"}))
    
    def test_heuristic_counts_missing_goals(self):
        """Heuristic returns number of goal facts not in state"""
        state = {"a", "b"}
        goals = {"a", "b", "c", "d"}
        
        self.assertEqual(heuristic(state, goals), 2)

class TestBFSSearch(unittest.TestCase):
    """BFS planning algorithm tests"""
    
    def test_bfs_finds_simple_plan(self):
        """BFS finds linear plan: a -> b -> c"""
        operators = [
            Operator("get_b", {"a"}, {"b"}),
            Operator("get_c", {"b"}, {"c"})
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        plan = solve_bfs(problem)
        
        self.assertIsNotNone(plan)
        self.assertEqual(plan.actions, ["get_b", "get_c"])
        self.assertEqual(plan.total_cost, 2.0)
    
    def test_bfs_prefers_fewer_actions(self):
        """BFS minimizes number of actions, not cost"""
        operators = [
            Operator("direct", {"a"}, {"c"}, cost=100.0),      # 1 action, expensive
            Operator("via_b", {"a"}, {"b"}, cost=1.0),         # 2 actions, cheap
            Operator("to_c", {"b"}, {"c"}, cost=1.0)
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        plan = solve_bfs(problem)
        
        # BFS chooses fewer actions (1) even though more expensive
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0], "direct")
    
    def test_bfs_no_solution(self):
        """BFS returns None when no plan exists"""
        operators = [Operator("get_b", {"a"}, {"b"})]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}  # Impossible to achieve
        )
        
        plan = solve_bfs(problem)
        
        self.assertIsNone(plan)

class TestAStarSearch(unittest.TestCase):
    """A* planning algorithm tests"""
    
    def test_astar_finds_simple_plan(self):
        """A* finds linear plan"""
        operators = [
            Operator("get_b", {"a"}, {"b"}),
            Operator("get_c", {"b"}, {"c"})
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        plan = solve_astar(problem)
        
        self.assertIsNotNone(plan)
        self.assertEqual(plan.actions, ["get_b", "get_c"])
    
    def test_astar_minimizes_cost(self):
        """A* finds plan with minimum total cost"""
        operators = [
            Operator("direct", {"a"}, {"c"}, cost=100.0),      # Expensive
            Operator("via_b", {"a"}, {"b"}, cost=1.0),         # Cheap path
            Operator("to_c", {"b"}, {"c"}, cost=1.0)
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        plan = solve_astar(problem)
        
        # A* chooses cheaper path (2 actions, cost 2) over expensive direct (cost 100)
        self.assertEqual(plan.total_cost, 2.0)
        self.assertEqual(plan.actions, ["via_b", "to_c"])
    
    def test_astar_with_heuristic(self):
        """A* uses heuristic to guide search"""
        operators = [
            Operator("wrong_path", {"start"}, {"mid1"}, cost=1.0),
            Operator("wrong_continue", {"mid1"}, {"mid2"}, cost=1.0),
            Operator("wrong_goal", {"mid2"}, {"goal"}, cost=100.0),
            Operator("right_path", {"start"}, {"goal"}, cost=2.0)
        ]
        problem = STRIPSProblem(
            propositions={"start", "mid1", "mid2", "goal"},
            operators=operators,
            initial_state={"start"},
            positive_goals={"goal"}
        )
        
        plan = solve_astar(problem)
        
        # Heuristic helps find cheaper path (cost 2) even though wrong path seems promising
        self.assertEqual(plan.total_cost, 2.0)

class TestRealWorldScenario(unittest.TestCase):
    """Real-world learning path scenario"""
    
    def test_web_development_path(self):
        """Planning a web development learning path"""
        # Courses with prerequisites
        operators = [
            Operator("HTML", set(), {"html"}, cost=1.0),
            Operator("CSS", {"html"}, {"css"}, cost=1.0),
            Operator("JavaScript", set(), {"javascript"}, cost=2.0),
            Operator("React", {"html", "css", "javascript"}, {"react"}, cost=3.0)
        ]
        
        problem = STRIPSProblem(
            propositions={"html", "css", "javascript", "react"},
            operators=operators,
            initial_state=set(),  # Knows nothing
            positive_goals={"react"}
        )
        
        plan = solve_astar(problem)
        
        self.assertIsNotNone(plan)
        self.assertIn("react", plan.final_state)
        
        # Verify React appears after its prerequisites
        react_index = plan.actions.index("React")
        prereqs_found = [op for op in plan.actions[:react_index] 
                        if op in ["HTML", "CSS", "JavaScript"]]
        self.assertEqual(len(prereqs_found), 3)
    
    def test_with_existing_skills(self):
        """User already has some skills"""
        operators = [
            Operator("HTML", set(), {"html"}, cost=1.0),
            Operator("React", {"html", "javascript"}, {"react"}, cost=3.0)
        ]
        
        # User already knows JavaScript
        problem = STRIPSProblem(
            propositions={"html", "javascript", "react"},
            operators=operators,
            initial_state={"javascript"},  # Already knows JS
            positive_goals={"react"}
        )
        
        plan = solve_astar(problem)
        
        self.assertIsNotNone(plan)
        # Only needs HTML, not JavaScript
        self.assertEqual(plan.actions, ["HTML", "React"])

class CompareBFSvsAStar(unittest.TestCase):
    """Compare BFS and A* behavior"""
    
    def test_same_for_unit_costs(self):
        """BFS and A* produce same plan when all costs are equal"""
        operators = [
            Operator("a_to_b", {"a"}, {"b"}, cost=1.0),
            Operator("b_to_c", {"b"}, {"c"}, cost=1.0),
            Operator("a_to_c", {"a"}, {"c"}, cost=1.0)
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        bfs_plan = solve_bfs(problem)
        astar_plan = solve_astar(problem)
        
        # Both find shortest plan (1 action)
        self.assertEqual(len(bfs_plan.actions), 1)
        self.assertEqual(len(astar_plan.actions), 1)
        self.assertEqual(bfs_plan.actions, astar_plan.actions)
    
    def test_different_when_costs_vary(self):
        """BFS and A* may differ when costs vary"""
        operators = [
            Operator("cheap_path", {"a"}, {"b"}, cost=1.0),
            Operator("cheap_goal", {"b"}, {"c"}, cost=1.0),
            Operator("expensive_direct", {"a"}, {"c"}, cost=100.0)
        ]
        problem = STRIPSProblem(
            propositions={"a", "b", "c"},
            operators=operators,
            initial_state={"a"},
            positive_goals={"c"}
        )
        
        bfs_plan = solve_bfs(problem)    # Prefers fewer actions (1)
        astar_plan = solve_astar(problem) # Prefers cheaper cost (2)
        
        self.assertEqual(len(bfs_plan.actions), 1)      # direct
        self.assertEqual(len(astar_plan.actions), 2)    # cheap_path + cheap_goal
        self.assertEqual(astar_plan.total_cost, 2.0)    # Cheaper total cost


if __name__ == "__main__":
    unittest.main(verbosity=2)
