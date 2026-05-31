"""Lightweight adapter for OpenAI-backed parsing and path explanations.

This module exposes helper functions to parse free-form user input into a
structured profile and to explain/compare candidate learning paths.
"""

import json
import re   # Offers regular expressions for pattern matching and text parsing
from typing import Dict, List, Tuple  # Type hints for dictionaries and lists
from certifi import contents
from openai import OpenAI   # OpenAI client for interacting with the OpenAI API
from pathlib import Path    # Path class for filesystem path manipulation
from typing import Optional # Optional type hint helper
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "deepseek/deepseek-chat-v3-0324"

def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()



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
    """Generate embedding for a single text using OpenAI API.
    
    Converts a text string into a 32-dimensional embedding vector using
    the text-embedding-3-small model via OpenRouter.
    
    Args:
        text: The text to embed
    
    Returns:
        List of floats representing the embedding, or None if generation fails
    """
    try:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=32
        )

        if res.data and len(res.data[0].embedding) > 0:
            return res.data[0].embedding
        
        return None
    
    except Exception as e:
        print(f"Error generating embedding for text '{text}': {e}")
        return None

def generate_embeddings(data_dir: Path) -> Optional[Path]:
    """Generate embeddings for all course titles and save to embedding.json."""

    try:
        # If embeddings already exist, skip regeneration to avoid unnecessary API calls.
        embedding_path = data_dir / "embedding.json"
        if embedding_path.exists():
            print(f"Skipping embeddings generation; {embedding_path.name} already exists.")
            return embedding_path.resolve()

        import json
        import time
        from openai import OpenAI

        # Read normalized_courses.json
        cache_path = data_dir / "normalized_courses.json"

        if not cache_path.exists():
            print(f"Error: {cache_path} not found")
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            courses = json.load(f)

        total_courses = len(courses)

        if total_courses == 0:
            print("No courses found in normalized_courses.json")
            return None

        embeddings_data = []

        # -------- CONFIG --------
        BATCH_SIZE = 500
        DIMENSIONS = 32
        MODEL = "text-embedding-3-small"
        # ------------------------

        start_time = time.perf_counter()

        # Process in batches
        for start_idx in range(0, total_courses, BATCH_SIZE):

            batch_courses = courses[start_idx:start_idx + BATCH_SIZE]

            # Extract course names
            batch_names = [
                course.get("name", "")
                for course in batch_courses
            ]

            try:
                # Single request for MANY titles
                res = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_names,
                    dimensions=32
                )

                # Save embeddings
                for course_name, item in zip(batch_names, res.data):

                    embeddings_data.append({
                        "name": course_name,
                        "embedding": item.embedding
                    })

            except Exception as e:
                print(
                    f"\nBatch error "
                    f"({start_idx}-{start_idx + len(batch_names)}): {e}"
                )
                continue

            # Progress
            processed = min(
                start_idx + BATCH_SIZE,
                total_courses
            )

            progress_percent = int(
                (processed / total_courses) * 100
            )

            print(
                f"\rProgress: "
                f"{progress_percent}% "
                f"({processed}/{total_courses})",
                end="",
                flush=True
            )

        # Save embeddings
        embedding_path = data_dir / "embedding.json"

        with open(
            embedding_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                embeddings_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        elapsed = (
            time.perf_counter() - start_time
        )

        print(
            f"\n✓ Embeddings saved to "
            f"{embedding_path}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

        return embedding_path.resolve()

    except Exception as e:
        print(
            f"Error in generate_embeddings: {e}"
        )
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



def _call_prereq_llm(batch: List[str]) -> Dict[str, List[str]]:
    courses_text = "\n".join(batch)

    prompt = (
        "Return ONLY JSON.\n"
        "Values must be arrays containing 1 to 4 prerequisites.\n"
        "Rules:\n"
        "- keys must match input exactly\n"
        "- values: array of 1 to 4 items\n"
        "- Use 3-4 prerequisites only when genuinely necessary\n"
        "- Avoid redundant or nearly identical prerequisites\n"
        "- no extra text\n\n"
        f"Courses:\n{courses_text}\n"
    )

    try:
        response = ask_llm(prompt)
        if not response:
            return {}

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            return {}

        return json.loads(response[start:end+1])

    except Exception:
        return {}

def build_prerequisite_graph(course_names: List[str]) -> Dict[str, List[str]]:
    """Fast parallel prerequisite inference with progress tracking."""

    if not course_names:
        return {}

    BATCH_SIZE = 8

    batches = [
        course_names[i:i + BATCH_SIZE]
        for i in range(0, len(course_names), BATCH_SIZE)
    ]

    total_batches = len(batches)
    total_courses = len(course_names)

    results: Dict[str, List[str]] = {}
    processed_batches = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_call_prereq_llm, b) for b in batches]
        
        for f in as_completed(futures):
            try:
                data = f.result()
                if data:
                    results.update(data)

            except Exception:
                pass

            # 🔥 progreso
            processed_batches += 1
            processed_courses = min(processed_batches * BATCH_SIZE, total_courses)

            percent = (processed_courses / total_courses) * 100

            print(
                f"\rProgress: {percent:.2f}% "
                f"({processed_courses}/{total_courses} courses)",
                end="",
                flush=True
            )

    print()  # newline final limpio

    return results

#print(build_prerequisite_graph(["Python for Data Science", "Introduction to Machine Learning", "Web Development with JavaScript"]))
#print(get_text_embedding("Machine Learning"))