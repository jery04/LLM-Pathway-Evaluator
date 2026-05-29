"""Lightweight adapter for OpenAI-backed parsing and path explanations.

This module exposes helper functions to parse free-form user input into a
structured profile and to explain/compare candidate learning paths.
"""

import re   # Offers regular expressions for pattern matching and text parsing
from typing import Dict, List, Tuple  # Type hints for dictionaries and lists
from openai import OpenAI # OpenAI client for interacting with the OpenAI API

# Use environment variable for the API key to avoid embedding secrets in source.
client = OpenAI(
    api_key='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',
    base_url='https://openrouter.ai/api/v1'
)

def ask_llm(prompt):

    response = client.chat.completions.create(
        model='deepseek/deepseek-chat',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response.choices[0].message.content

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
