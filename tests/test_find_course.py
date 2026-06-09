"""
Unit tests for the course search and skill matching functionality.

This test suite validates the find_course_by_skill function and its dependencies:
- load_courses: Loads course data from source
- build_skill_index: Creates a searchable index of skills from courses
- _load_embeddings_data: Loads precomputed embeddings for semantic similarity
- find_course_by_skill: Main function that finds and ranks courses by skill match

The tests verify that find_course_by_skill returns valid results for various skills,
including proper result counts, non-empty course names, and valid similarity scores
between 0 and 1.
"""

import sys          # System module for path manipulation
import os           # OS module for file/directory operations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))  # Add src directory to Python path
import unittest     # Unit testing framework

# Import the functions to be tested 
from planner import load_courses, build_skill_index, _load_embeddings_data, find_course_by_skill  


class TestCourseSearch(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Validate that courses is not empty
        self.courses = load_courses()
        self.assertIsNotNone(self.courses, "courses is None")
        self.assertNotEqual(len(self.courses), 0, "courses is empty")
        
        # Validate that skill_index is not empty
        self.skill_index = build_skill_index(self.courses)
        self.assertIsNotNone(self.skill_index, "skill_index is None")
        self.assertNotEqual(len(self.skill_index), 0, "skill_index is empty")
        
        # Validate that embeddings_data is not empty
        self.embeddings_data = _load_embeddings_data()
        self.assertIsNotNone(self.embeddings_data, "embeddings_data is None")
        
        self.test_skills = [
            "machine learning",
            "python",
            "data science",
            "deep learning",
            "web development",
        ]
    
    def test_find_course_by_skill(self):
        """Test that find_course_by_skill returns valid results"""
        
        for skill in self.test_skills:
            with self.subTest(skill=skill):
                results = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=5
                )
                
                # Validate that results is not None or empty
                self.assertIsNotNone(results, f"No results for {skill}")
                self.assertNotEqual(len(results), 0, f"Empty results list for {skill}")
                self.assertEqual(len(results), 5, f"Should be 5 results for {skill}, got {len(results)}")
                
                for i, r in enumerate(results):
                    # Validate that name is not empty
                    self.assertIn('name', r, f"Result {i} missing 'name' for {skill}")
                    self.assertIsNotNone(r['name'], f"Name is None for result {i} of {skill}")
                    self.assertNotEqual(r['name'].strip(), "", f"Empty name for result {i} of {skill}")
                    
                    # Validate that similarity_score exists and is a positive number
                    self.assertIn('similarity_score', r, f"Result {i} missing 'similarity_score' for {skill}")
                    self.assertIsNotNone(r['similarity_score'], f"similarity_score is None for result {i} of {skill}")
                    self.assertIsInstance(r['similarity_score'], (int, float), f"similarity_score is not a number for {skill}")
                    self.assertGreater(r['similarity_score'], 0, f"similarity_score must be > 0 for {skill}, got {r['similarity_score']}")
                    
                    # Additional validation: score should be <= 1
                    self.assertLessEqual(r['similarity_score'], 1, f"similarity_score > 1 for {skill}: {r['similarity_score']}")
    
    def test_find_course_by_skill_empty_skill(self):
        """Test with empty skill string"""
        results = find_course_by_skill(
            skill="",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        self.assertEqual(results, [], "Empty skill should return empty list")
    
    def test_find_course_by_skill_none_skill(self):
        """Test with None skill"""
        results = find_course_by_skill(
            skill=None,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        self.assertEqual(results, [], "None skill should return empty list")
    
    def test_find_course_by_skill_whitespace_skill(self):
        """Test with whitespace-only skill"""
        results = find_course_by_skill(
            skill="   ",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        self.assertEqual(results, [], "Whitespace skill should return empty list")
    
    def test_find_course_by_skill_top_n_variations(self):
        """Test different top_n values"""
        skills_to_test = ["python", "java", "sql"]
        
        for skill in skills_to_test:
            with self.subTest(skill=skill):
                # Test top_n=1
                results_1 = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=1
                )
                self.assertIsNotNone(results_1)
                self.assertLessEqual(len(results_1), 1)
                
                # Test top_n=3
                results_3 = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=3
                )
                self.assertIsNotNone(results_3)
                self.assertLessEqual(len(results_3), 3)
                
                # Test top_n=10
                results_10 = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=10
                )
                self.assertIsNotNone(results_10)
                self.assertLessEqual(len(results_10), 10)
    
    def test_find_course_by_skill_scores_decreasing(self):
        """Test that similarity scores are in decreasing order"""
        skill = "machine learning"
        
        results = find_course_by_skill(
            skill=skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        
        # Check that scores are in descending order
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]['similarity_score'], 
                results[i + 1]['similarity_score'],
                f"Scores not decreasing: {results[i]['similarity_score']} < {results[i + 1]['similarity_score']}"
            )
    
    def test_find_course_by_skill_no_duplicates(self):
        """Test that there are no duplicate courses in results"""
        skill = "data science"
        
        results = find_course_by_skill(
            skill=skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=10
        )
        
        course_names = [r['name'] for r in results]
        self.assertEqual(len(course_names), len(set(course_names)), "Duplicate courses found in results")
    
    def test_find_course_by_skill_case_insensitivity(self):
        """Test that skill search is case insensitive"""
        lower_case = find_course_by_skill(
            skill="python",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=3
        )
        
        upper_case = find_course_by_skill(
            skill="PYTHON",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=3
        )
        
        mixed_case = find_course_by_skill(
            skill="PyThOn",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=3
        )
        
        # Results should be similar regardless of case
        self.assertEqual(len(lower_case), len(upper_case))
        self.assertEqual(len(lower_case), len(mixed_case))
    
    def test_find_course_by_skill_partial_match(self):
        """Test partial skill matching (e.g., 'ml' for 'machine learning')"""
        full_skill = find_course_by_skill(
            skill="machine learning",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=3
        )
        
        partial_skill = find_course_by_skill(
            skill="ml",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=3
        )
        
        # Both should return results
        self.assertGreater(len(full_skill), 0)
        self.assertGreater(len(partial_skill), 0)
    
    def test_find_course_by_skill_nonexistent_skill(self):
        """Test with a completely made-up skill"""
        results = find_course_by_skill(
            skill="completely_nonexistent_skill_xyz_123",
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        
        # May return empty list or some fallback results
        self.assertIsNotNone(results)
        # Should still be a valid list (could be empty)
        self.assertIsInstance(results, list)
    
    def test_find_course_by_skill_valid_scores_range(self):
        """Test that all similarity scores are between 0 and 1"""
        skills_to_test = ["python", "java", "sql", "aws", "docker"]
        
        for skill in skills_to_test:
            with self.subTest(skill=skill):
                results = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=5
                )
                
                for r in results:
                    score = r['similarity_score']
                    self.assertGreaterEqual(score, 0, f"Score < 0 for {skill}: {score}")
                    self.assertLessEqual(score, 1, f"Score > 1 for {skill}: {score}")
    
    def test_find_course_by_skill_returns_valid_names(self):
        """Test that all returned course names exist in the course index"""
        from planner import index_courses
        
        course_index = index_courses(self.courses)
        skill = "python"
        
        results = find_course_by_skill(
            skill=skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=10
        )
        
        for r in results:
            course_name = r['name']
            self.assertIn(course_name, course_index, f"Course '{course_name}' not found in course index")
    
    def test_find_course_by_skill_reproducibility(self):
        """Test that multiple calls with same parameters return same results"""
        skill = "deep learning"
        
        results1 = find_course_by_skill(
            skill=skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        
        results2 = find_course_by_skill(
            skill=skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        
        # Results should be identical
        self.assertEqual(results1, results2)
    
    def test_find_course_by_skill_with_special_characters(self):
        """Test skills with special characters"""
        special_skills = ["c++", "c#", "f#", ".net", "node.js"]
        
        for skill in special_skills:
            with self.subTest(skill=skill):
                results = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=3
                )
                # Should not crash, may return empty or valid results
                self.assertIsInstance(results, list)
    
    def test_find_course_by_skill_with_numbers(self):
        """Test skills that contain numbers"""
        numeric_skills = ["python3", "c++11", "web2py", "django2"]
        
        for skill in numeric_skills:
            with self.subTest(skill=skill):
                results = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=3
                )
                self.assertIsInstance(results, list)
    
    def test_find_course_by_skill_very_long_skill_name(self):
        """Test with an extremely long skill name"""
        long_skill = "extremely_long_skill_name_" * 50
        
        results = find_course_by_skill(
            skill=long_skill,
            skill_index=self.skill_index,
            embeddings_data=self.embeddings_data,
            top_n=5
        )
        
        # Should handle gracefully (return empty list or few results)
        self.assertIsInstance(results, list)
    
    def test_find_course_by_skill_multiple_word_skill(self):
        """Test multi-word skills"""
        multi_word_skills = [
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "big data analytics",
            "cloud computing architecture"
        ]
        
        for skill in multi_word_skills:
            with self.subTest(skill=skill):
                results = find_course_by_skill(
                    skill=skill,
                    skill_index=self.skill_index,
                    embeddings_data=self.embeddings_data,
                    top_n=3
                )
                self.assertIsInstance(results, list)
                
                if len(results) > 0:
                    for r in results:
                        self.assertIn('name', r)
                        self.assertIn('similarity_score', r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
