"""
Unit tests for planner utility functions.

This test suite validates lower-level helper functions used by the course
planning system, including token normalization, safe numeric conversion,
metric extraction, and course scoring logic.
"""

import sys    # System module for path manipulation
import os     # OS module for file/directory operations

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest                      # Unit testing framework
import unittest                      # Unit testing framework
from planner import (
    _normalize_token,                # Helper: normalizes text tokens
    _safe_float,                     # Helper: safely converts to float
    _course_metrics,                 # Helper: computes course metrics
    _score_course_for_criterion,     # Helper: scores a course against a criterion
)


class TestPlannerUtilities(unittest.TestCase):
    """Tests for planner utility functions and criterion scoring."""
    
    def setUp(self):
        """Set up sample course data for utility tests."""
        self.course = {
            "name": "Python Basics",
            "duration_months": 2,
            "difficulty": 3,
            "cost_usd": 49.99,
        }
        self.invalid_course = {
            "name": "Broken Course",
            "duration_months": "two",
            "difficulty": None,
            "cost_usd": "free",
        }
        self.high_cost_course = {
            "name": "Expensive Course",
            "duration_months": 20,
            "difficulty": 20,
            "cost_usd": 1200,
        }
    
    def test_normalize_token_trims_and_lowercases(self):
        """Normalize string tokens by trimming whitespace and lowering case."""
        self.assertEqual(_normalize_token('  PyThOn  '), 'python')
        self.assertEqual(_normalize_token('   '), '')
        self.assertEqual(_normalize_token(None), '')
    
    def test_normalize_token_empty_string(self):
        """Empty or whitespace-only strings normalize to an empty token."""
        self.assertEqual(_normalize_token(''), '')
        self.assertEqual(_normalize_token('   '), '')
        self.assertEqual(_normalize_token('\t\n'), '')
    
    def test_safe_float_returns_float_for_numeric_values(self):
        """Convert numeric strings and numbers to float values."""
        self.assertEqual(_safe_float('12.5'), 12.5)
        self.assertEqual(_safe_float(8), 8.0)
        self.assertEqual(_safe_float(0), 0.0)
        self.assertEqual(_safe_float('0'), 0.0)
    
    def test_safe_float_returns_default_on_invalid_input(self):
        """Return the provided fallback when conversion is impossible."""
        self.assertEqual(_safe_float('abc', default=7.0), 7.0)
        self.assertEqual(_safe_float(None, default=-1.0), -1.0)
        self.assertEqual(_safe_float([], default=2.5), 2.5)
        self.assertEqual(_safe_float({}, default=0.5), 0.5)
    
    def test_course_metrics_extracts_numeric_metrics(self):
        """Extract numeric metrics and coerce values to float."""
        self.assertEqual(_course_metrics(self.course), {
            'duration_months': 2.0,
            'cost_usd': 49.99,
            'difficulty': 3.0,
        })
    
    def test_course_metrics_converts_string_numbers(self):
        """Coerce numeric strings to floats when extracting metrics."""
        course = {
            'duration_months': '4',
            'cost_usd': '199.99',
            'difficulty': '7',
        }
        self.assertEqual(_course_metrics(course), {
            'duration_months': 4.0,
            'cost_usd': 199.99,
            'difficulty': 7.0,
        })
    
    def test_course_metrics_uses_defaults_for_invalid_values(self):
        """Use default metric values when conversion fails or values are missing."""
        self.assertEqual(_course_metrics(self.invalid_course), {
            'duration_months': 1.0,
            'cost_usd': 0.0,
            'difficulty': 3.0,
        })
        self.assertEqual(_course_metrics({'name': 'Empty Course'}), {
            'duration_months': 1.0,
            'cost_usd': 0.0,
            'difficulty': 3.0,
        })
    
    def test_score_course_for_cheapest_path(self):
        """Compute cheapest-path scores with cost as the dominant factor."""
        score = _score_course_for_criterion(self.course, 'Cheapest path')
        expected = 0.80 * (49.99 / 500.0) + 0.12 * (2.0 / 6.0) + 0.08 * (3.0 / 10.0)
        self.assertAlmostEqual(score, expected, places=6)
    
    def test_score_course_for_fastest_path(self):
        """Compute fastest-path scores with duration as the dominant factor."""
        score = _score_course_for_criterion(self.course, 'Fastest path')
        expected = 0.80 * (2.0 / 6.0) + 0.18 * (3.0 / 10.0) + 0.02 * (49.99 / 500.0)
        self.assertAlmostEqual(score, expected, places=6)
    
    def test_score_course_for_balanced_path(self):
        """Compute balanced-path scores as a weighted average of all metrics."""
        score = _score_course_for_criterion(self.course, 'Balanced path')
        expected = 0.45 * (2.0 / 6.0) + 0.35 * (49.99 / 500.0) + 0.20 * (3.0 / 10.0)
        self.assertAlmostEqual(score, expected, places=6)
    
    def test_score_course_caps_metric_values_at_one(self):
        """Ensure overly large metric values are normalized to the [0,1] range."""
        score = _score_course_for_criterion(self.high_cost_course, 'Cheapest path')
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)
    
    def test_score_course_defaults_to_balanced_for_unknown_criterion(self):
        """Unknown criteria should fall back to the balanced scoring formula."""
        unknown_score = _score_course_for_criterion(self.course, 'Unknown')
        balanced_score = _score_course_for_criterion(self.course, 'Balanced path')
        self.assertEqual(unknown_score, balanced_score)
    
    def test_score_course_rank_order_is_consistent_for_criteria(self):
        """Higher-cost courses should score worse under cheapest-path criteria."""
        cheap_course = {
            'duration_months': 2,
            'difficulty': 3,
            'cost_usd': 10,
        }
        expensive_course = {
            'duration_months': 2,
            'difficulty': 3,
            'cost_usd': 400,
        }
        cheap_score = _score_course_for_criterion(cheap_course, 'Cheapest path')
        expensive_score = _score_course_for_criterion(expensive_course, 'Cheapest path')
        self.assertLess(cheap_score, expensive_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
