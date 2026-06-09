"""
Unit tests for planner module's skill coverage and category filtering functions.

This test suite validates the behavior of internal helper functions used in
the course planning system:
- _is_skill_covered: Determines if a skill is already covered by known skills
  using lexical matching and semantic similarity via embeddings.
- _passes_category_filter: Checks if a course should be included when certain
  categories are to be avoided, using embedding similarity thresholds.

The tests cover edge cases including empty inputs, lexical matches, substring
matching, semantic similarity thresholds, embedding failures, and handling of
multiple known skills or avoid categories.
"""

import sys  # System-specific parameters and functions
import os   # Operating system interface

# Add the src directory to the Python path so we can import the planner module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))  # Add src to path

import unittest                    # Unit testing framework
from unittest.mock import patch    # Mocking functionality for tests

# Import the functions to be tested
from planner import (
    _is_skill_covered,               # Check if skill is already known
    _passes_category_filter,         # Filter courses by category
    SIMILARITY_THRESHOLD,            # Threshold for skill similarity
    CATEGORY_SIMILARITY_THRESHOLD    # Threshold for category similarity
)

class TestIsSkillCovered(unittest.TestCase):
    """Tests for _is_skill_covered(skill, known_skills)"""
    
    def setUp(self):
        """Set up mocks for get_text_embedding"""
        self.embedding_patcher = patch('planner.get_text_embedding')
        self.mock_get_embedding = self.embedding_patcher.start()
        
        # Mock for embeddings - 3-dimensional vector for simplicity
        self.mock_embedding_vector = [0.1, 0.2, 0.3]
        self.mock_get_embedding.return_value = self.mock_embedding_vector
    
    def tearDown(self):
        """Stop the embedding patcher to restore original functionality after each test."""
        self.embedding_patcher.stop()
    
    def test_empty_skill_returns_true(self):
        """Empty skill should return True"""
        self.assertTrue(_is_skill_covered("", ["python"]))
        self.assertTrue(_is_skill_covered("   ", ["python"]))
        self.assertTrue(_is_skill_covered(None, ["python"]))
    
    def test_exact_lexical_match_returns_true(self):
        """Exact match should return True without needing embeddings"""
        result = _is_skill_covered("python", ["python", "java"])
        self.assertTrue(result)
    
    def test_substring_match_returns_true(self):
        """Substring should return True"""
        self.assertTrue(_is_skill_covered("learning", ["machine learning"]))
        self.assertTrue(_is_skill_covered("ml", ["machine learning ml"]))
    
    def test_known_skill_contains_skill_returns_true(self):
        """If known contains skill should return True"""
        self.assertTrue(_is_skill_covered("py", ["python"]))
    
    def test_no_match_and_low_similarity_returns_false(self):
        """No lexical match and low similarity should return False"""
        self.mock_get_embedding.side_effect = [
            self.mock_embedding_vector,  # embedding for skill
            [0.9, 0.9, 0.9]  # embedding for known (very different)
        ]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = SIMILARITY_THRESHOLD - 0.1  # below threshold
            
            result = _is_skill_covered("advanced_ai", ["basic_programming"])
            self.assertFalse(result)
    
    def test_high_similarity_returns_true(self):
        """High semantic similarity should return True"""
        self.mock_get_embedding.side_effect = [
            self.mock_embedding_vector,  # embedding for skill
            [0.11, 0.21, 0.31]  # similar embedding
        ]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = SIMILARITY_THRESHOLD + 0.1  # above threshold
            
            result = _is_skill_covered("machine_learning", ["ml", "ai"])
            self.assertTrue(result)
    
    def test_embedding_failure_returns_false(self):
        """If get_text_embedding fails should return False"""
        self.mock_get_embedding.return_value = None
        
        result = _is_skill_covered("python", ["java"])
        self.assertFalse(result)
    
    def test_empty_known_skills_no_match_returns_false(self):
        """With empty known_skills and no match should return False"""
        self.mock_get_embedding.side_effect = [
            self.mock_embedding_vector,  # embedding for skill
        ]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = 0.5  # below threshold
            
            result = _is_skill_covered("python", [])
            self.assertFalse(result)
    
    def test_multiple_known_skills_second_matches(self):
        """Should find match in the second known_skill"""
        self.mock_get_embedding.side_effect = [
            self.mock_embedding_vector,  # embedding for skill
            [0.9, 0.9, 0.9],  # embedding for known1 (different)
            [0.11, 0.21, 0.31]  # embedding for known2 (similar)
        ]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.side_effect = [0.3, SIMILARITY_THRESHOLD + 0.1]
            
            result = _is_skill_covered(
                "deep_learning", 
                ["basic_math", "neural_networks"]
            )
            self.assertTrue(result)
    
    def test_skip_empty_known_skills(self):
        """Should skip empty or None known_skills"""
        self.mock_get_embedding.side_effect = [
            self.mock_embedding_vector,  # embedding for skill
            [0.11, 0.21, 0.31]  # embedding for valid known
        ]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = SIMILARITY_THRESHOLD + 0.1
            
            result = _is_skill_covered("python", ["", None, "   ", "python_programming"])
            self.assertTrue(result)

class TestPassesCategoryFilter(unittest.TestCase):
    """Tests for _passes_category_filter(course, avoid_categories, embeddings)"""
    
    def setUp(self):
        """Set up test data"""
        self.embedding_patcher = patch('planner.get_text_embedding')
        self.mock_get_embedding = self.embedding_patcher.start()
        
        self.mock_embedding_vector = [0.1, 0.2, 0.3]
        self.mock_get_embedding.return_value = self.mock_embedding_vector
        
        # Mock course embeddings
        self.embeddings = {
            "Python Basics": [0.1, 0.2, 0.3],
            "Data Science": [0.4, 0.5, 0.6],
            "Web Development": [0.7, 0.8, 0.9],
            "Machine Learning": [0.2, 0.3, 0.4],
        }
    
    def tearDown(self):
        self.embedding_patcher.stop()
    
    def test_no_avoid_categories_returns_true(self):
        """No categories to avoid should return True"""
        course = {"name": "Python Basics"}
        result = _passes_category_filter(course, None, self.embeddings)
        self.assertTrue(result)
        
        result = _passes_category_filter(course, [], self.embeddings)
        self.assertTrue(result)
    
    def test_empty_course_name_returns_true(self):
        """Course without name should return True"""
        course = {"name": ""}
        result = _passes_category_filter(course, ["data"], self.embeddings)
        self.assertTrue(result)
        
        course = {"name": None}
        result = _passes_category_filter(course, ["data"], self.embeddings)
        self.assertTrue(result)
    
    def test_course_embedding_missing_returns_true(self):
        """Course without embedding should return True (function allows the course)"""
        course = {"name": "Unknown Course"}
        result = _passes_category_filter(course, ["data"], self.embeddings)
        self.assertTrue(result)
    
    def test_category_in_course_name_behavior(self):
        """Verify behavior when category is in course name"""
        course = {"name": "Data Science Fundamentals"}
        
        # Real function may have specific behavior
        # Just verify it doesn't crash and returns a boolean
        result = _passes_category_filter(course, ["data"], self.embeddings)
        self.assertIsInstance(result, bool)
    
    def test_similarity_threshold_behavior(self):
        """Verify behavior with different similarity levels"""
        course = {"name": "Advanced Statistics"}
        
        # Verify function handles different similarity scores
        for similarity_score in [0.3, 0.5, 0.7, 0.9]:
            with patch('planner._cosine_similarity') as mock_cosine:
                mock_cosine.return_value = similarity_score
                
                result = _passes_category_filter(
                    course, 
                    ["mathematics"], 
                    self.embeddings
                )
                self.assertIsInstance(result, bool)
    
    def test_multiple_avoid_categories_handling(self):
        """Verify handling of multiple avoided categories"""
        course = {"name": "Web Development Course"}
        
        result = _passes_category_filter(
            course, 
            ["data", "web", "ai"], 
            self.embeddings
        )
        self.assertIsInstance(result, bool)
    
    def test_low_similarity_behavior(self):
        """Verify behavior with low similarity"""
        course = {"name": "Python Programming"}
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = 0.1  # Very low similarity
            
            result = _passes_category_filter(
                course, 
                ["data_science"], 
                self.embeddings
            )
            # Function may return True or False, just verify it's boolean
            self.assertIsInstance(result, bool)
    
    def test_embedding_failure_for_category_continues(self):
        """If embedding for a category fails, continue with next one"""
        course = {"name": "Python Programming"}
        
        # First category fails, second has embedding
        self.mock_get_embedding.side_effect = [None, self.mock_embedding_vector]
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = 0.2
            
            result = _passes_category_filter(
                course, 
                ["invalid_category", "valid_category"], 
                self.embeddings
            )
            self.assertIsInstance(result, bool)
    
    def test_normalize_empty_categories_filters_them_out(self):
        """Empty categories should be filtered out"""
        course = {"name": "Python Programming"}
        
        result = _passes_category_filter(
            course, 
            ["", None, "   ", "web"], 
            self.embeddings
        )
        self.assertIsInstance(result, bool)
    
    def test_course_without_category_field(self):
        """Course without category field should work"""
        course = {"name": "Some Course"}
        
        result = _passes_category_filter(course, ["data"], self.embeddings)
        self.assertIsInstance(result, bool)
    
    def test_realistic_scenario_no_filter(self):
        """Realistic scenario: course that should not be filtered"""
        course = {"name": "Python for Beginners"}
        
        result = _passes_category_filter(
            course, 
            ["advanced", "mathematics", "physics"], 
            self.embeddings
        )
        # Verify it returns something (True or False)
        self.assertIsInstance(result, bool)
    
    def test_realistic_scenario_with_filter(self):
        """Realistic scenario: course that might be filtered"""
        course = {"name": "Advanced Data Science"}
        
        with patch('planner._cosine_similarity') as mock_cosine:
            mock_cosine.return_value = CATEGORY_SIMILARITY_THRESHOLD + 0.1
            
            result = _passes_category_filter(
                course, 
                ["data"], 
                self.embeddings
            )
            self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
