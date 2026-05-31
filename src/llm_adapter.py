"""Lightweight adapter for OpenAI-backed parsing and path explanations.

This module exposes helper functions to parse free-form user input into a
structured profile and to explain/compare candidate learning paths.
"""

import json
import re   # Offers regular expressions for pattern matching and text parsing
from typing import Dict, List, Tuple  # Type hints for dictionaries and lists
from openai import OpenAI   # OpenAI client for interacting with the OpenAI API
from pathlib import Path    # Path class for filesystem path manipulation
from typing import Optional # Optional type hint helper
from google import genai
import time

client = genai.Client(
    api_key="AQ.Ab8RN6LByPrhqnpqPPFaMiKRcAKJoO_D1CIh2IowDdYZk96p2g"
)

MODEL = "gemini-2.5-flash"

def ask_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()


#--------------------------------------------------------------------
# METHODS
#--------------------------------------------------------------------

def parse_input(text: str) -> Tuple[str, List[str], List[str]]:
    """Parse free-form user text and return (goal, preferences, skills).

    The LLM is asked to return plain text (no JSON). Expected outputs:
    - Goal: single-line with the main goal (2-5 words)
    - Preferences: comma-separated list on one line
    - Skills: comma-separated list on one line
    """

    # Extraer goal - objetivo principal en 2-5 palabras
    goal_prompt = (
        f"Extract the main learning goal in 2-5 words.\n"
        f"Respond in English. Output ONLY the goal on a single line, nothing else.\n\n"
        f"Text: {text}"
    )
    goal_response = ask_llm(goal_prompt)
    goal = goal_response.strip().splitlines()[0] if goal_response else 'Not specified'

    # Extraer preferences - como lista separada por comas
    prefs_prompt = (
        f"Extract all user preferences and requirements as a comma-separated list.\n"
        f"Respond in English. Output ONLY one line with items separated by commas (e.g. 'learn fast, visual, project-based').\n\n"
        f"Text: {text}"
    )
    prefs_response = ask_llm(prefs_prompt) or ''
    # Split on common delimiters and the word 'and'
    raw_prefs = re.split(r',|;|\n|\band\b', prefs_response)
    preferences = [p.strip() for p in raw_prefs if p and p.strip()]

    # Extraer skills - como lista separada por comas
    skills_prompt = (
        f"Extract existing skills, tools, and technologies the user already knows as a comma-separated list.\n"
        f"Respond in English. Output ONLY one line with items separated by commas.\n\n"
        f"Text: {text}"
    )
    skills_response = ask_llm(skills_prompt) or ''
    raw_skills = re.split(r',|;|\n|\band\b', skills_response)
    skills = [s.strip() for s in raw_skills if s and s.strip()]

    return (goal or 'Not specified', preferences, skills)

def explain_paths_brief(paths: List[Dict]) -> List[str]:
    """Return a short LLM-written explanation for each path."""
    brief = []

    for i, p in enumerate(paths, 1):
        criterion = p.get('criterion') or f'Path {i}'
        path = p.get('path', [])
        metrics = p.get('metrics', {})
        route = ' -> '.join(path) or 'No courses'

        prompt = (
            'Write a brief, natural, precise explanation in English for this path. '
            'Use at most 2-5 sentences. '
            'Do not use lists or a mechanical tone. '
            'Include the overall progression and a small practical insight if it fits. Respond in English.\n\n'
            f'Path: {criterion}\n'
            f'Courses: {route}\n'
            f'Time: {metrics.get("total_months", "?")} months\n'
            f'Cost: ${metrics.get("total_cost", "?")}\n'
            f'Average difficulty: {metrics.get("avg_difficulty", "?")}\n'
            f'Steps: {metrics.get("steps", "?")}'
        )

        response = ask_llm(prompt)
        brief.append((response or '').strip())

    return brief

def explain_comparison(paths: List[Dict], user_profile: Dict) -> str:
    """Produce an LLM-written comparison and final recommendation."""
    profile_parts = []
    for key in ('goal', 'objective'):
        value = user_profile.get(key)
        if value:
            profile_parts.append(f'{key}: {value}')
    if user_profile.get('skills'):
        profile_parts.append(f"skills: {', '.join(user_profile['skills'])}")
    if user_profile.get('preferences'):
        profile_parts.append(f"preferences: {', '.join(user_profile['preferences'])}")

    path_lines = []
    for i, p in enumerate(paths, 1):
        criterion = p.get('criterion') or f'Path {i}'
        route = ' -> '.join(p.get('path', [])) or 'No courses'
        metrics = p.get('metrics', {})
        path_lines.append(
            f'{i}. {criterion} | {route} | time={metrics.get("total_months", "?")} months | '
            f'cost=${metrics.get("total_cost", "?")} | difficulty={metrics.get("avg_difficulty", "?")} | steps={metrics.get("steps", "?")}'
        )

    prompt = (
        'Write a clear, natural comparison in English. '
        'Do not use tables or a mechanical tone. '
        'For each path, explain in 1-2 sentences what it offers and when it makes sense. '
        'Close with one concrete, brief recommendation. Respond in English.\n\n'
        f'User profile: {"; ".join(profile_parts) if profile_parts else "no extra data"}\n\n'
        f'Paths:\n{chr(10).join(path_lines)}'
    )

    response = ask_llm(prompt)
    return (response or '').strip()

def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text using Gemini API.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding, or None if it fails
    """
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

        # Gemini devuelve embeddings aquí:
        return response.embeddings[0].values

    except Exception as e:
        print(f"Error generating embedding for text '{text}': {e}")
        return None

def generate_embeddings(data_dir: Path) -> Optional[Path]:
    """Generate embeddings for all course titles and save to embedding.json."""

    try:
        embedding_path = data_dir / "embedding.json"

        # Skip if already exists
        if embedding_path.exists():
            print(f"Skipping embeddings generation; {embedding_path.name} already exists.")
            return embedding_path.resolve()

        cache_path = data_dir / "normalized_courses.json"

        if not cache_path.exists():
            print(f"Error: {cache_path} not found")
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            courses = json.load(f)

        if not courses:
            print("No courses found in normalized_courses.json")
            return None

        embeddings_data = []

        BATCH_SIZE = 100  # Gemini suele ser más sensible que OpenAI
        MODEL = "gemini-embedding-001"

        start_time = time.perf_counter()

        for start_idx in range(0, len(courses), BATCH_SIZE):

            batch = courses[start_idx:start_idx + BATCH_SIZE]

            batch_names = [
                course.get("name", "").strip()
                for course in batch
                if course.get("name")
            ]

            try:
                # GEMINI EMBEDDINGS CALL
                response = client.models.embed_content(
                    model=MODEL,
                    contents=batch_names
                )

                # response.embeddings -> lista en el mismo orden
                for course_name, emb in zip(batch_names, response.embeddings):

                    embeddings_data.append({
                        "name": course_name,
                        "embedding": emb.values
                    })

            except Exception as e:
                print(f"\nBatch error ({start_idx}-{start_idx + len(batch_names)}): {e}")
                continue

            processed = min(start_idx + BATCH_SIZE, len(courses))
            progress = int((processed / len(courses)) * 100)

            print(
                f"\rProgress: {progress}% ({processed}/{len(courses)})",
                end="",
                flush=True
            )

        with open(embedding_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - start_time

        print(f"\n✓ Embeddings saved to {embedding_path}")
        print(f"Time: {elapsed:.2f}s")

        return embedding_path.resolve()

    except Exception as e:
        print(f"Error in generate_embeddings: {e}")
        return None

def infer_prerequisites_for_objective(objective: str) -> List[str]:
    """Infer prerequisites needed to learn a given objective or skill.
    
    Uses LLM to determine what foundational knowledge or skills are required
    before tackling the specified objective.
    
    Args:
        objective: The learning objective or skill to analyze (e.g., "Machine Learning", "Web Development")
    
    Returns:
        List of prerequisites in English (e.g., ["Python", "Linear Algebra", "Statistics"])
    """
    
    prompt = (
        "You are an expert educational advisor. Given a learning objective, "
        "identify the essential prerequisites needed to master it effectively.\n\n"
        "Objective: " + objective + "\n\n"
        "Provide a comma-separated list of 3-5 key prerequisites that a learner should have "
        "BEFORE starting to learn this objective. Be specific and practical.\n"
        "Respond in English with ONLY the prerequisites list, nothing else.\n"
        "Example format: 'Python basics, Linear Algebra, Statistics fundamentals'\n"
    )
    
    try:
        response = ask_llm(prompt)
        
        # Parse comma-separated prerequisites
        if response:
            prerequisites = [p.strip() for p in response.split(',') if p.strip()]
            return prerequisites
        
        return []
    
    except Exception as e:
        print(f"Error inferring prerequisites for objective '{objective}': {e}")
        return []

