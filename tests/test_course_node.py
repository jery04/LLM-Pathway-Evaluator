"""
Unit tests for CourseNode dataclass and its methods.

This test suite validates the tree structure functionality for prerequisite
course resolution, including flattening and printing operations.
"""

import sys                          # System module for path manipulation
import os                           # OS module for file/directory operations
from io import StringIO             # Captures printed output for testing

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))  # Add src to path for imports

import unittest                     # Unit testing framework
from planner import CourseNode      # Import CourseNode dataclass to test


class TestCourseNode(unittest.TestCase):
    """Tests for CourseNode dataclass and static methods"""
    
    def test_create_empty_node(self):
        """Should create a node with only name"""
        node = CourseNode(name="Python 101")
        
        self.assertEqual(node.name, "Python 101")
        self.assertIsNone(node.skill)
        self.assertEqual(node.alternatives, [])
        self.assertEqual(node.unresolved, [])
        self.assertEqual(node.children, [])
    
    def test_create_node_with_all_fields(self):
        """Should create a node with all fields populated"""
        children = [
            CourseNode(name="Prerequisite 1"),
            CourseNode(name="Prerequisite 2")
        ]
        
        node = CourseNode(
            name="Advanced Python",
            skill="python",
            alternatives=["Course A", "Course B"],
            unresolved=["regex", "decorators"],
            children=children
        )
        
        self.assertEqual(node.name, "Advanced Python")
        self.assertEqual(node.skill, "python")
        self.assertEqual(node.alternatives, ["Course A", "Course B"])
        self.assertEqual(node.unresolved, ["regex", "decorators"])
        self.assertEqual(len(node.children), 2)
        self.assertEqual(node.children[0].name, "Prerequisite 1")
    
    def test_flatten_single_node(self):
        """Flatten should return list with single node name"""
        node = CourseNode(name="Single Course")
        
        result = CourseNode.flatten(node)
        
        self.assertEqual(result, ["Single Course"])
    
    def test_flatten_node_with_one_child(self):
        """Flatten should return child first, then parent (post-order)"""
        child = CourseNode(name="Prerequisite")
        parent = CourseNode(name="Main Course", children=[child])
        
        result = CourseNode.flatten(parent)
        
        self.assertEqual(result, ["Prerequisite", "Main Course"])
    
    def test_flatten_node_with_multiple_children(self):
        """Flatten should return all children (post-order) before parent"""
        child1 = CourseNode(name="Prereq 1")
        child2 = CourseNode(name="Prereq 2")
        child3 = CourseNode(name="Prereq 3")
        parent = CourseNode(name="Main", children=[child1, child2, child3])
        
        result = CourseNode.flatten(parent)
        
        # All children should come before parent
        self.assertEqual(result, ["Prereq 1", "Prereq 2", "Prereq 3", "Main"])
    
    def test_flatten_nested_tree(self):
        """Flatten should handle nested tree structure correctly (post-order)"""
        # Build tree:
        # Level 3 (deepest)
        leaf1 = CourseNode(name="Leaf 1")
        leaf2 = CourseNode(name="Leaf 2")
        
        # Level 2
        middle1 = CourseNode(name="Middle 1", children=[leaf1])
        middle2 = CourseNode(name="Middle 2", children=[leaf2])
        
        # Level 1 (root)
        root = CourseNode(name="Root", children=[middle1, middle2])
        
        result = CourseNode.flatten(root)
        
        # Expected post-order: leaf1, middle1, leaf2, middle2, root
        expected = ["Leaf 1", "Middle 1", "Leaf 2", "Middle 2", "Root"]
        self.assertEqual(result, expected)
    
    def test_flatten_complex_tree(self):
        """Flatten should handle complex tree with multiple levels"""
        # Create deeper tree
        leaf = CourseNode(name="Leaf")
        middle = CourseNode(name="Middle", children=[leaf])
        top = CourseNode(name="Top", children=[middle])
        root = CourseNode(name="Root", children=[top])
        
        result = CourseNode.flatten(root)
        
        expected = ["Leaf", "Middle", "Top", "Root"]
        self.assertEqual(result, expected)
    
    def test_flatten_node_with_no_children_returns_self(self):
        """Flatten on leaf node should return just its name"""
        node = CourseNode(name="Leaf Node")
        
        result = CourseNode.flatten(node)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Leaf Node")
    
    def test_flatten_does_not_modify_original(self):
        """Flatten should not modify the original tree structure"""
        child = CourseNode(name="Child")
        parent = CourseNode(name="Parent", children=[child])
        
        original_children_count = len(parent.children)
        
        CourseNode.flatten(parent)
        
        self.assertEqual(len(parent.children), original_children_count)
        self.assertEqual(parent.children[0].name, "Child")
    
    def test_print_tree_single_node(self):
        """Print tree should output just the node name"""
        node = CourseNode(name="Only Course")
        
        captured_output = StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(node)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.assertIn("📘 Only Course", output)
    
    def test_print_tree_with_skill(self):
        """Print tree should show skill when present"""
        node = CourseNode(name="Python Course", skill="python")
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(node)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.assertIn("📘 Python Course", output)
        self.assertIn("🎯 skill: python", output)
    
    def test_print_tree_with_alternatives(self):
        """Print tree should show alternatives when present"""
        node = CourseNode(
            name="Data Science",
            alternatives=["Course A", "Course B", "Course C"]
        )
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(node)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.assertIn("📘 Data Science", output)
        self.assertIn("🔁 alternatives: ['Course A', 'Course B', 'Course C']", output)
    
    def test_print_tree_with_unresolved(self):
        """Print tree should show unresolved skills when present"""
        node = CourseNode(
            name="Advanced ML",
            unresolved=["linear algebra", "calculus", "statistics"]
        )
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(node)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.assertIn("📘 Advanced ML", output)
        self.assertIn("⚠️ unresolved: ['linear algebra', 'calculus', 'statistics']", output)
    
    def test_print_tree_with_children(self):
        """Print tree should show children with indentation"""
        child1 = CourseNode(name="Prereq 1")
        child2 = CourseNode(name="Prereq 2")
        parent = CourseNode(name="Main Course", children=[child1, child2])
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(parent)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.assertIn("📘 Main Course", output)
        self.assertIn("    📘 Prereq 1", output)  # Indented with 4 spaces
        self.assertIn("    📘 Prereq 2", output)
    
    def test_print_tree_nested_with_indentation(self):
        """Print tree should increase indentation for deeper levels"""
        leaf = CourseNode(name="Leaf")
        middle = CourseNode(name="Middle", children=[leaf])
        root = CourseNode(name="Root", children=[middle])
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(root)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        lines = output.strip().split('\n')
        
        # Root should have no indentation
        self.assertTrue(lines[0].startswith("📘 Root"))
        # Middle should have 4 spaces indentation
        self.assertTrue(lines[1].startswith("    📘 Middle"))
        # Leaf should have 8 spaces indentation
        self.assertTrue(lines[2].startswith("        📘 Leaf"))
    
    def test_print_tree_with_all_features(self):
        """Print tree should handle nodes with all features combined"""
        child = CourseNode(
            name="Child Course",
            skill="child_skill",
            alternatives=["Alt 1", "Alt 2"],
            unresolved=["missing_skill"]
        )
        
        parent = CourseNode(
            name="Parent Course",
            skill="parent_skill",
            alternatives=["Alternative X"],
            unresolved=["unresolved_x", "unresolved_y"],
            children=[child]
        )
        
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            CourseNode.print_tree(parent)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Check parent features
        self.assertIn("📘 Parent Course", output)
        self.assertIn("🎯 skill: parent_skill", output)
        self.assertIn("🔁 alternatives: ['Alternative X']", output)
        self.assertIn("⚠️ unresolved: ['unresolved_x', 'unresolved_y']", output)
        
        # Check child features (with indentation)
        self.assertIn("    📘 Child Course", output)
        self.assertIn("    🎯 skill: child_skill", output)
        self.assertIn("    🔁 alternatives: ['Alt 1', 'Alt 2']", output)
        self.assertIn("    ⚠️ unresolved: ['missing_skill']", output)
    
    def test_flatten_preserves_order_for_different_paths(self):
        """Flatten should maintain consistent order for different branches"""
        # Left branch
        left_leaf = CourseNode(name="Left Leaf")
        left_branch = CourseNode(name="Left Branch", children=[left_leaf])
        
        # Right branch
        right_leaf = CourseNode(name="Right Leaf")
        right_branch = CourseNode(name="Right Branch", children=[right_leaf])
        
        root = CourseNode(name="Root", children=[left_branch, right_branch])
        
        result = CourseNode.flatten(root)
        
        # Order should preserve children order: left branch first, then right branch
        expected = ["Left Leaf", "Left Branch", "Right Leaf", "Right Branch", "Root"]
        self.assertEqual(result, expected)
    
    def test_dataclass_equality(self):
        """Two nodes with same data should be equal"""
        node1 = CourseNode(name="Course", skill="python", alternatives=["A"], unresolved=["B"])
        node2 = CourseNode(name="Course", skill="python", alternatives=["A"], unresolved=["B"])
        
        self.assertEqual(node1, node2)
    
    def test_dataclass_inequality(self):
        """Nodes with different data should not be equal"""
        node1 = CourseNode(name="Course A")
        node2 = CourseNode(name="Course B")
        
        self.assertNotEqual(node1, node2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
