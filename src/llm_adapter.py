"""Gemini adapter for user profiling, path explanations, and course embeddings.

- Parse free text into structured learning preferences and skills.
- Explain and compare candidate learning paths via LLM.
- Infer prerequisites for any learning objective.
- Generate spaCy embeddings (EN/ES) for course titles.

Uses Google Gemini (gemini-2.5-flash-lite) and pre-loaded spaCy models.
"""

from dotenv import load_dotenv # Load environment variables from .env file for API keys and config
import os   # Operating system interactions for file paths and environment variables
import json # JSON serialization/deserialization for cache and data exchange
import re   # Regular expressions for parsing LLM responses
from typing import Dict, List, Tuple  # Type hints for dictionaries, lists, and tuples
from pathlib import Path     # Object-oriented filesystem path manipulation
from typing import Optional  # Type hint for optional values (None or T)
from google import genai     # Google Gemini API client for LLM interactions
import time   # Performance measurement and progress tracking (used in generate_embeddings)
import spacy  # NLP library for text embeddings and language detection fallback
from langdetect import detect  # Language detection to choose spacy model (es/en)

# Load variables from the .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Global Gemini client and model configuration
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Model choice - can be switched to a more advanced model if needed
MODEL = "gemini-3.1-flash-lite-preview"

# Pre-load spaCy models for English and Spanish
nlp_es = spacy.load('es_core_news_md')  # Spanish model
nlp_en = spacy.load('en_core_web_md')   # English model


# ===========================================================================
# LANGUAGE, MODEL PICKER & BASIC LLM INTERFACE
# ===========================================================================

def pick_model(texto):
    """Select the appropriate spaCy language model based on detected text language.
    This returns the Spanish model for Spanish text and the English model otherwise.
    """
    try:
        idioma = detect(texto)
        if idioma == 'es':
            return nlp_es
        else:
            return nlp_en
    except:
        print(f"⚠️ Could not detect language of '{texto}', using EN model by default")
        return nlp_en

def ask_llm(prompt: str) -> str:
    """Send a prompt to the Gemini model and return the stripped text response.
    The result is returned as plain text with leading and trailing whitespace removed.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    return response.text.strip()


# ===========================================================================
# LLM PARSER & EXTRACTOR 
# ===========================================================================

def parse_input(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """Parse a free-form learning request into goal, preferences, skills, and avoids.
    The function returns a structured profile extracted from the user text.
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
        """Extract JSON text from model output whether fenced or plain.
        The helper returns the first JSON object found in the response text.
        """
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
    """Generate short LLM explanations for each candidate path in relation to the goal.
    Each explanation links the path details to the stated learning objective.
    """
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
    """Write an LLM comparison of candidate paths and return one brief recommendation.
    The function summarizes differences and suggests the best path for the user.
    """
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


# ===========================================================================
# EMBEDDINGS GENERATION
# ===========================================================================

def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate a spaCy text embedding for a single input using language detection.
    The embedding is produced by a model selected for the detected text language.
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
    """Create and save spaCy embeddings for course titles, skipping existing files.
    The results are stored in embedding.json and reused if already present.
    """
    
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


# ===========================================================================
# PREREQUISITE INFERENCE 
# ===========================================================================

def infer_prerequisites_for_objective(objective: str) -> List[str]:
    """Use the LLM to infer whether a learning objective needs prior knowledge.
    The function returns a short list of prerequisites if they are truly required.
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

