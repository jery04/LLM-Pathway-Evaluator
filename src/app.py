"""Streamlit web UI glue for the Career Path Explorer app.

This module wires the local planner and LLM adapter into a lightweight
Streamlit interface. It handles dynamic local imports, form inputs, and
renders course links and generated learning paths.
"""

import streamlit as st          # Web UI framework for building interactive apps
from typing import List         # Type hint for lists
from pathlib import Path        # Object‑oriented filesystem paths
import importlib.util           # Utilities for loading modules dynamically
import types                    # Provides dynamic type creation and inspection
from html import escape         # Escapes HTML to prevent injection
from urllib.parse import quote_plus   # URL‑encodes strings for safe query parameters
from planner import index_courses, load_courses, generate_paths  # Local planner functions
from llm_adapter import parse_input, explain_comparison, explain_paths_brief  # Local LLM adapter functions


def _merge_skill_lists(*skill_groups: List[str]) -> List[str]:
    """Merge skill lists while preserving order and removing duplicates."""
    merged = []
    seen = set()

    for group in skill_groups:
        for skill in group or []:
            normalized = str(skill or '').strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(str(skill).strip())

    return merged

def _course_link(course: dict) -> str:
    """Return a best-effort URL for a course dict.

    Prefer explicit link fields; otherwise produce a platform-specific
    search URL or a Google search if platform is unknown.
    """
    # Check several common fields that might contain a direct link.
    link = str(course.get('link') or course.get('url') or course.get('href') or '').strip()
    if link:
        return link

    # If no direct link, build a search URL using the course title.
    title = str(course.get('name')).strip()
    if not title:
        return 'https://www.google.com/search'

    platform = str(course.get('platform', '')).strip().lower()
    # Map known platform slugs to their search URLs for better UX.
    if platform == 'edx':
        return f'https://www.edx.org/search?q={quote_plus(title)}'
    if platform == 'skillshare':
        return f'https://www.skillshare.com/search?query={quote_plus(title)}'
    if platform == 'udemy':
        return f'https://www.udemy.com/courses/search/?q={quote_plus(title)}'
    if platform == 'coursera':
        return f'https://www.coursera.org/search?query={quote_plus(title)}'

    # Fallback to a general Google search for the course title.
    return f'https://www.google.com/search?q={quote_plus(title)}'

def _render_course_buttons(path: List[str], course_index: dict) -> None:
    """Render HTML links for each course in the provided path list.

    Looks up course metadata in `course_index` and outputs a styled
    block of anchor tags. Uses `unsafe_allow_html=True` intentionally
    but escapes URL and label parts to avoid injection.
    """
    # Gather course objects for the path and remove missing entries.
    courses = [course_index.get(name) for name in path]
    courses = [course for course in courses if course]
    if not courses:
        return

    links = []
    for course in courses:
        # Prefer English `name`, fall back to Spanish `nombre`.
        label = str(course.get('name') or course.get('nombre') or 'Open course')
        link = _course_link(course)
        # Escape both URL and label to avoid HTML injection attacks.
        links.append(
            f'<a class="course-link-word" href="{escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
        )

    # Render the inline links using a styled container defined in the page CSS.
    st.markdown(f'<div class="course-links">{"".join(links)}</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit app: layout, input handling, and path rendering.

    Orchestrates loading data, parsing user input, generating candidate
    paths, and showing qualitative LLM comparisons.
    """
    
    st.set_page_config(page_title='Career Path Explorer')
    st.markdown(
        """
        <style>
        .hero-header {
            min-height: 16vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            text-align: center;
            margin-bottom: 1.0rem;
            padding-top: 0.25rem;
        }
        .hero-header h1 {
            margin: 0;
            font-size: clamp(2.0rem, 3.6vw, 3.2rem);
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .hero-header p {
            margin: 0.05rem 0 0;
            max-width: 46rem;
            color: rgba(200, 200, 200, 0.9);
            font-size: 1.0rem;
            line-height: 1.35;
        }
        .course-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem 0.6rem;
            margin-top: 0.75rem;
            margin-bottom: 0.35rem;
        }
        .course-link-word {
            display: inline-block;
            color: #4a78b8 !important;
            text-decoration: underline;
            font-weight: 600;
            font-size: 0.95rem;
            line-height: 1.35;
        }
        .course-link-word:hover {
            color: #37639e !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render the app hero header and description using embedded CSS/HTML.
    st.markdown(
        """
        <div class="hero-header">
            <h1>Career Path Explorer</h1>
            <p>Generate, validate, and compare alternative career paths under optimization criteria.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load course catalog and build quick lookup structures.
    courses = load_courses()
    course_index = index_courses(courses)

    # Choose a sensible default selection for the skills multiselect UI.
    default_skills = ['python basics']
    # Build a form to capture the user's goal, existing skills, and optimization criterion.
    with st.form('input_form'):
        user_text = st.text_area('Describe your goal and constraints (eg: "I want to learn data science while avoiding math")', height=120)
        initial_skills = st.multiselect('Skills you already have', options=course_index.keys(), default=default_skills)
        selected_criterion = st.radio(
            'Optimization criteria',
            options=['Fastest path', 'Cheapest path', 'Balanced path'],
            index=0,
            horizontal=True,
            help='Each criterion produces a different path to compare time, cost, and difficulty.'
        )
        submitted = st.form_submit_button('Generate paths')

    if submitted:
        # If the user submits an empty goal, show a simple message and stop early.
        if not (user_text and user_text.strip()):
            st.info('No goal entered.')
            return

        # Parse free-form user text into a structured profile (goal, preferences, skills, avoids).
        goal, preferences, parser_skills, parser_avoids = parse_input(user_text or '')
        user_skills = _merge_skill_lists(initial_skills, parser_skills)
        selected_criteria = [selected_criterion] if selected_criterion else ['Fastest path']

        # Generate candidate paths using the planner. This may be CPU-bound.
        with st.spinner('Generating paths...'):
            knows_python = any('python' in skill.lower() for skill in user_skills)
            paths = generate_paths(
                courses,
                user_skills,
                goal,
                max_paths=max(1, len(selected_criteria)),
                avoid_categories=parser_avoids,
                user_prefs={'knows_python': knows_python},
                criteria_names=selected_criteria,
            )

        # Show detected goal and skills to the user for transparency.
        st.caption(f'Detected goal: {goal or "not detected"}')

        if not paths:
            st.warning('No valid paths were found. Adjust the goal, constraints, or existing skills.')
        else:
            # Get brief explanations for each path from the LLM helper (optional).
            brief_explanations = []
            try:
                brief_explanations = explain_paths_brief(paths)
            except Exception:
                # If the LLM helper fails, the UI can still render the raw paths.
                brief_explanations = []

            # Render each path with metrics, explanation, and course links.
            for i, p in enumerate(paths, 1):
                target = p.get('target_course') or goal
                criterion = p.get('criterion') or f'Path {i}'
                st.subheader(f"{criterion}: {' → '.join(p['path'])}")
                # show the target course text directly in gray (no label)
                if target:
                    st.caption(target)
                m = p['metrics']
                st.caption(f"Total time: {m['total_months']} months | Cost: ${m['total_cost']} | Average difficulty: {m['avg_difficulty']}")

                # show brief multi-line explanation in gray if available
                if i - 1 < len(brief_explanations):
                    st.caption(brief_explanations[i - 1])
                _render_course_buttons(p['path'], course_index)

            # Qualitative comparison generated by the LLM helper (long-form).
            st.subheader('Qualitative comparison (LLM)')
            try:
                explanation = explain_comparison(paths, {'goal': goal, 'skills': user_skills, 'objective': goal, 'preferences': preferences, 'avoids': parser_avoids})
            except Exception:
                explanation = ''
            st.text_area('Explanation', value=explanation, height=240)


if __name__ == '__main__':
    main()
