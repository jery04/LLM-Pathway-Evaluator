"""Experimental script to validate the quality of the system's responses.

This script is a test harness designed to evaluate the quality of learning path generation
using a controlled environment: a specific dataset (experiments.json), predefined parameters,
and three optimization strategies (cheapest, fastest, balanced)

Key Features:
    1. Loads course data from experiments.json
    2. Generates vector embeddings for course titles via Spacy
    3. Produces and compares 3 pathfinding strategies (cheapest, fastest, balanced)
    
Use Case:
    Quality validation of the planning system before production deployment.
"""

from pathlib import Path    # Import Path class for cross-platform filesystem path handling
import json                 # Import JSON module for parsing and writing JSON data
import time                 # Import time module for sleep, timestamps, or performance measurement
from typing import List, Dict, Optional  # Import type hints for better code documentation and IDE support

# Add the workspace root to sys.path so scripts executed from subdirectories can import local modules.
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent.parent / 'src'))

from llm_adapter import get_text_embedding  # Import function to convert text into vector embeddings using LLM
from planner import generate_paths          # Import function to generate possible routes/solutions from the planner module


# ===========================================================================
# LOAD & EMBEDDING GENERATION
# ===========================================================================

def load_experiments() -> List[Dict]:
    """Load experiments from experiments/experiments.json.
    
    Returns a list of experiment dictionaries or an empty list on error.
    """

    json_path = Path(__file__).resolve().parent / "experiments.json"

    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {json_path}: {e}")
        return []

def generate_experiment_embeddings(experiments: List[Dict], output_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate embeddings for each experiment and save to embedding.json.
    
    Returns the path to the written file, or None on failure.
    """

    base_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    embedding_path = base_dir / "embedding.json"

    # If embeddings file already exists, skip work and return its path
    if embedding_path.exists():
        print(f"Embeddings already exist at {embedding_path}, skipping generation.")
        return embedding_path

    if not experiments:
        print("No experiments found to embed")
        return None

    embeddings: List[Dict] = []
    start_time = time.perf_counter()
    total = len(experiments)

    for i, exp in enumerate(experiments, 1):
        name = (exp.get("name") or exp.get("title") or "").strip()
        if not name:
            continue

        try:
            vector = get_text_embedding(name)
        except Exception as e:
            print(f"⚠️ Error generating embedding for '{name}': {e}")
            vector = None

        if vector is not None:
            embeddings.append({"name": name, "embedding": vector})

        print(f"\r{i}/{total} ({i*100//total}%)", end="", flush=True)

    try:
        with open(embedding_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - start_time
        print(f"\n✅ Saved {len(embeddings)} embeddings in {elapsed:.1f}s: {embedding_path}")
        return embedding_path

    except Exception as e:
        print(f"⚠️ Error writing embeddings to {embedding_path}: {e}")
        return None


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """Main function - Experimental script to generate learning paths."""
    
    print("🧪 EXPERIMENTAL MODE: Generating learning paths")
    print("=" * 70)
    
    # Load courses from the normalized dataset
    courses = load_experiments()
    
    if not courses:
        print("❌ No courses loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(courses)} courses")
    print()
    
    # Generate embeddings for courses
    print("📊 Generating embeddings...")
    generate_experiment_embeddings(courses)
    print()
    
    initial_skills: List[str] = []  # Starting with no initial skills
    objective: str = "learn data science"  # Goal
    max_paths: int = 2  # Generate up to 2 paths
    avoid_categories: Optional[List[str]] = None  # No categories to avoid
    user_prefs: Optional[List[str]] = None  # No preferences
    criterion_name: Optional[str] = 'Cheapest path'  # 'Fastest path' Use default criterion
    experiment: bool = True  # Enable experiment mode
    
    print("📋 Parameters:")
    print(f"   - goal: '{objective}'")
    print(f"   - experiment: {experiment}")
    print()

    criteria_options = [
        ("cheap", "Cheapest path"),
        ("fast", "Fastest path"),
        ("balanced", "Balanced path"),
    ]

    for criteria_label, criterion_name in criteria_options:
        paths = generate_paths(
            courses=courses,
            initial_skills=initial_skills,
            objective=objective,
            max_paths=max_paths,
            avoid_categories=avoid_categories,
            user_prefs=user_prefs,
            criterion_name=criterion_name,
            experiment=experiment
        )
        
        print()
        print("=" * 70)
        print(f"✅ Generated {len(paths)} learning paths for criterion '{criteria_label}'")
        print("=" * 70)
        print()
        
        if paths:
            for idx, path in enumerate(paths, 1):
                print(f"📌 Path {idx}:")
                print(f"   Target Course: {path.get('target_course', 'N/A')}")
                
                steps = path.get('steps', []) or path.get('course_path', [])
                if steps:
                    course_sequence = " → ".join(steps)
                    print(f"   📚 Courses: {course_sequence}")
                
                metrics = path.get('metrics', {})
                if metrics:
                    print(f"   Total Steps: {metrics.get('steps', 0)}")
                    print(f"   Total Duration: {metrics.get('total_months', 0)} months")
                    print(f"   Total Cost: ${metrics.get('total_cost', 0):.2f}")
                    print(f"   Average Difficulty: {metrics.get('avg_difficulty', 0):.2f}")
                print()
        else:
            print(f"⚠️  No learning paths were generated for criterion '{criteria_label}'.")
            print()

if __name__ == "__main__":
    main()
