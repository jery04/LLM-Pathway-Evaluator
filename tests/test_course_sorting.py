"""
Unit tests for _sort_courses_by_criterion function.

This test suite validates the sorting behavior of courses based on different
criteria (price, rating, duration, etc.) used in the course planning system.
"""

import sys    # System module for path manipulation
import os     # OS module for file/directory operations

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest                                # Unit testing framework
from planner import _sort_courses_by_criterion  # Function to sort courses by optimization criteria


class TestSortCoursesByCriterion(unittest.TestCase):
    """Tests for _sort_courses_by_criterion(course_names, course_index, criterion_name)"""
    
    def setUp(self):
        """Set up test course data and index"""
        self.courses = [
            {"name": "Python Basics", "price": 49.99, "rating": 4.5, "duration_months": 40, "difficulty": 3, "cost_usd": 49.99},
            {"name": "Advanced Python", "price": 89.99, "rating": 4.8, "duration_months": 60, "difficulty": 6, "cost_usd": 89.99},
            {"name": "Data Science", "price": 99.99, "rating": 4.7, "duration_months": 80, "difficulty": 7, "cost_usd": 99.99},
            {"name": "Web Dev", "price": 39.99, "rating": 4.3, "duration_months": 50, "difficulty": 4, "cost_usd": 39.99},
            {"name": "Machine Learning", "price": 129.99, "rating": 4.9, "duration_months": 100, "difficulty": 8, "cost_usd": 129.99},
        ]
        
        # Build course index
        self.course_index = {course["name"]: course for course in self.courses}
        self.course_names = list(self.course_index.keys())
    
    def test_sort_by_cheapest_path_criterion(self):
        """Should sort courses using 'Cheapest path' scoring"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        # Cheapest path minimizes cost + 0.35*duration + 0.2*difficulty
        # Web Dev (39.99 + 0.35*50 + 0.2*4 = 39.99 + 17.5 + 0.8 = 58.29) should be first
        self.assertEqual(sorted_names[0], "Web Dev")
    
    def test_sort_by_fastest_path_criterion(self):
        """Should sort courses using 'Fastest path' scoring"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Fastest path"
        )
        
        # Fastest path minimizes duration + 0.001*cost + 0.25*difficulty
        # Python Basics (40 + 0.05 + 0.75 = 40.8) should be first
        self.assertEqual(sorted_names[0], "Python Basics")
    
    def test_sort_by_balanced_path_criterion(self):
        """Should sort courses using default 'Balanced path' scoring"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Balanced path"
        )
        
        # Balanced: 0.5*duration + 0.002*cost + 0.4*difficulty
        # Should return valid sorted list
        self.assertEqual(len(sorted_names), 5)
    
    def test_sort_by_unknown_criterion(self):
        """Should fall back to 'Balanced path' for unknown criterion"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Unknown criterion"
        )
        
        # Should still return all courses
        self.assertEqual(len(sorted_names), 5)
    
    def test_empty_course_names_list(self):
        """Should return empty list when course_names is empty"""
        result = _sort_courses_by_criterion([], self.course_index, "Cheapest path")
        self.assertEqual(result, [])
    
    def test_course_name_not_in_index(self):
        """Should skip course names not found in index"""
        names_with_missing = ["Web Dev", "Nonexistent Course", "Python Basics"]
        
        sorted_names = _sort_courses_by_criterion(
            names_with_missing, 
            self.course_index, 
            "Cheapest path"
        )
        
        # Should only include valid courses
        self.assertEqual(len(sorted_names), 2)
        self.assertIn("Web Dev", sorted_names)
        self.assertIn("Python Basics", sorted_names)
    
    def test_single_course(self):
        """Should handle single course correctly"""
        single_name = ["Python Basics"]
        
        result = _sort_courses_by_criterion(
            single_name, 
            self.course_index, 
            "Cheapest path"
        )
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Python Basics")
    
    def test_all_courses_returned(self):
        """Should return all valid course names (not necessarily all input)"""
        result = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        # Should return all 5 courses
        self.assertEqual(len(result), 5)
        
        # Should contain all original names
        for name in self.course_names:
            self.assertIn(name, result)
    
    def test_scores_are_consistent(self):
        """Same courses should sort consistently with same criterion"""
        result1 = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        result2 = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        # Should return same order
        self.assertEqual(result1, result2)
    
    def test_different_criteria_give_different_orders(self):
        """Different optimization criteria should produce different rankings"""
        cheapest = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        fastest = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Fastest path"
        )
        
        # Orders should be different
        self.assertNotEqual(cheapest, fastest)
    
    def test_case_insensitive_criterion(self):
        """Should handle criterion names with different case"""
        result1 = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "cheapest path"
        )
        
        result2 = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "CHEAPEST PATH"
        )
        
        # Should return same order (case insensitive)
        self.assertEqual(result1, result2)
    
    def test_realistic_scenario_cheapest_first(self):
        """Realistic scenario: cheapest path should prioritize low cost courses"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Cheapest path"
        )
        
        # Web Dev (cost 39.99) and Python Basics (49.99) should be top
        top_courses = sorted_names[:2]
        self.assertIn("Web Dev", top_courses)
        self.assertIn("Python Basics", top_courses)
        
        # Most expensive (Machine Learning at 129.99) should be last
        self.assertEqual(sorted_names[-1], "Machine Learning")
    
    def test_realistic_scenario_fastest_first(self):
        """Realistic scenario: fastest path should prioritize short duration"""
        sorted_names = _sort_courses_by_criterion(
            self.course_names, 
            self.course_index, 
            "Fastest path"
        )
        
        # Python Basics (40 months) should be first
        self.assertEqual(sorted_names[0], "Python Basics")


if __name__ == "__main__":
    unittest.main(verbosity=2)
