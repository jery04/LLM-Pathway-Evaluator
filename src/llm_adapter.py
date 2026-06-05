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
import spacy
from scipy.spatial.distance import cosine
import numpy as np
import langdetect
from langdetect import detect


client = genai.Client(
    api_key="AQ.Ab8RN6KQjo57zJ4VkQ57HYRV5tuEjAfUpwkCdBwd4Qki-0RAXw"
)

MODEL = "gemini-2.5-flash-lite"


# Cargar ambos modelos al inicio
nlp_es = spacy.load('es_core_news_md')  # modelo en español
nlp_en = spacy.load('en_core_web_md')   # modelo en inglés

# Función para seleccionar el modelo según el idioma
def pick_model(texto):
    try:
        idioma = detect(texto)
        if idioma == 'es':
            return nlp_es
        else:
            return nlp_en
    except:
        print(f"⚠️ No se pudo detectar el idioma de '{texto}', usando modelo EN por defecto")
        return nlp_en

def ask_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    return response.text.strip()


#--------------------------------------------------------------------
# METHODS
#--------------------------------------------------------------------

def parse_input(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Parse free-form user text and return:
    (goal, preferences, skills, avoids)
    """

    prompt = f"""
        Extract information from the user's learning request.
        Return ONLY valid JSON.

        Schema:
        {{
            "goal": "main learning goal in 2-5 words",
            "preferences": ["desired approaches, tools, technologies, requirements"],
            "skills": ["skills, tools, technologies, languages, frameworks already known"],
            "avoids": ["topics, technologies, methods, requirements, or subjects the user wants to avoid"]
        }}

        Rules:
        - Respond in English.
        - goal must be concise (2-5 words).
        - preferences must contain only positive preferences.
        - Do NOT put existing skills in preferences.
        - Do NOT put avoided topics in preferences.
        - skills must contain only things the user already knows.
        - avoids must contain only things the user wants to avoid, minimize, skip, or not focus on.
        - If a field has no values, return an empty array.
        - Return JSON only. No markdown. No explanations.

        User text:
        {text}
    """

    response = ask_llm(prompt) or ""

    def _extract_json_payload(raw_text: str) -> str:
        """Extract a JSON object from fenced or plain model output."""
        cleaned_text = raw_text.strip()

        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.IGNORECASE | re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()

        start_index = cleaned_text.find("{")
        end_index = cleaned_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return cleaned_text[start_index:end_index + 1].strip()

        return cleaned_text

    try:
        data = json.loads(_extract_json_payload(response))

        goal = str(data.get("goal", "Not specified")).strip()

        preferences = [
            str(x).strip()
            for x in data.get("preferences", [])
            if str(x).strip()
        ]

        skills = [
            str(x).strip()
            for x in data.get("skills", [])
            if str(x).strip()
        ]

        avoids = [
            str(x).strip()
            for x in data.get("avoids", [])
            if str(x).strip()
        ]

        return goal, preferences, skills, avoids

    except Exception:
        return "Not specified", [], [], []

def explain_paths_brief(paths: List[Dict], goal: str = '') -> List[str]:
    """Return a short LLM-written explanation for each path, including how it helps the user's goal."""
    brief = []

    for i, p in enumerate(paths, 1):
        criterion = p.get('criterion') or f'Path {i}'
        path = p.get('path', [])
        metrics = p.get('metrics', {})
        route = ' -> '.join(path) or 'No courses'

        prompt = (
            'Write 1-3 short sentences in English. '
            'Mention how this path helps achieve the user goal. '
            'Do not use lists or a mechanical tone.\n\n'
            f'Goal: {goal or "unknown"}\n'
            f'Path: {criterion}\n'
            f'Courses: {route}\n'
            f'Time: {metrics.get("total_months", "?")} months\n'
            f'Cost: ${metrics.get("total_cost", "?")}\n'
            f'Average difficulty: {metrics.get("avg_difficulty", "?")}\n'
        )

        response = ask_llm(prompt)
        brief.append((response or '').strip())

    return brief

def explain_comparison(paths: List[Dict], user_profile: Dict) -> str:
    """Produce an LLM-written comparison and final recommendation."""
    profile_parts = []
    if user_profile.get('goal'):
        profile_parts.append(f'goal: {user_profile.get("goal")}')
    if user_profile.get('skills'):
        profile_parts.append(f"skills: {', '.join(user_profile['skills'])}")
    if user_profile.get('preferences'):
        profile_parts.append(f"preferences: {', '.join(user_profile['preferences'])}")
    if user_profile.get('avoids'):
        profile_parts.append(f"avoids: {', '.join(user_profile['avoids'])}")

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
        'For each path, explain in 1-2 sentences: what does the path offer that others do not?'
        'Close with one concrete, brief recommendation. Respond in English.\n\n'
        f'User profile: {"; ".join(profile_parts) if profile_parts else "no extra data"}\n\n'
        f'Paths:\n{chr(10).join(path_lines)}'
    )

    response = ask_llm(prompt)
    return (response or '').strip()

def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text using spaCy model based on language detection.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding, or None if it fails
    """
    try:
        # Seleccionar el modelo según el idioma
        nlp = pick_model(text)
        
        # Procesar el texto con el modelo seleccionado
        doc = nlp(text)
        return doc.vector.tolist()
        
    except Exception as e:
        print(f"⚠️ Error processing text '{text}': {e}")
        return None

def generate_embeddings(data_dir: Path) -> Optional[Path]:
    """Generate embeddings for all course titles using spaCy."""
    
    embedding_path = data_dir / "embedding.json"
    
    # Skip if already exists
    if embedding_path.exists():
        print(f"Embeddings already exist at {embedding_path}")
        return embedding_path
    
    # Load courses - ¡AÑADIR encoding="utf-8"!
    courses_path = data_dir / "normalized_courses.json"
    if not courses_path.exists():
        print(f"Error: {courses_path} not found")
        return None
    
    with open(courses_path, "r", encoding="utf-8") as f:  # <- UTF-8 obligatorio
        courses = json.load(f)
    
    if not courses:
        print("No courses found")
        return None
    
    # Generate embeddings
    embeddings_data = []
    start_time = time.perf_counter()
    total = len(courses)
    
    for i, course in enumerate(courses, 1):
        name = course.get("name", "").strip()
        if not name:
            continue
            
        vector = get_text_embedding(name)
        if vector is not None:
            embeddings_data.append({"name": name, "embedding": vector})

        # Progress
        print(f"\r{i}/{total} ({i*100//total}%)", end="", flush=True)
    
    # Save results - también con utf-8
    with open(embedding_path, "w", encoding="utf-8") as f:  # <- UTF-8 también al escribir
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Saved {len(embeddings_data)} embeddings in {elapsed:.1f}s")
    
    return embedding_path

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
        "identify IF and ONLY IF the objective genuinely requires prior knowledge. "
        "Return 1-3 specific prerequisites when truly necessary. "
        "If the objective is basic enough that any person could learn it without prior knowledge, "
        "return NOTHING (empty response).\n\n"
        "Objective: " + objective + "\n\n"
        "Rules:\n"
        "- Return MAXIMUM 3 prerequisites, never more\n"
        "- Return NOTHING if the topic is beginner-friendly and requires no prior knowledge\n"
        "- Respond with ONLY the comma-separated prerequisites list, OR nothing at all\n"
        "- No explanations, no extra words\n\n"
        "Examples:\n"
        "Objective: 'Deep Learning' -> 'Machine Learning basics, Python programming, Linear Algebra'\n"
        "Objective: 'Basic Cooking' -> \n"
        "Objective: 'Calculus' -> 'Algebra, Trigonometry'\n"
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

