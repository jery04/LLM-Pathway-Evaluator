import streamlit as st
from typing import List
from pathlib import Path
import importlib.util
import types
from html import escape
from urllib.parse import quote_plus


def _load_local_module(module_name: str, rel_path: str) -> types.ModuleType:
    """Load a module from the repository file to guarantee we import local code.

    This avoids accidentally importing third-party packages named the same
    (for example `planner`) when the app is run with a different interpreter
    or via `streamlit run` outside the virtualenv.
    """
    repo_root = Path(__file__).resolve().parent
    file_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Always load the local modules by file path to be robust across environments.
_planner = _load_local_module('local_planner', 'planner.py')
_llm = _load_local_module('local_llm_adapter', 'llm_adapter.py')

load_courses = _planner.load_courses
generate_paths = _planner.generate_paths
parse_input = _llm.parse_input
explain_comparison = _llm.explain_comparison

def load_data():
    return load_courses()


def _course_link(course: dict) -> str:
    link = str(course.get('link') or course.get('url') or course.get('href') or '').strip()
    if link:
        return link

    title = str(course.get('name')).strip()
    if not title:
        return 'https://www.google.com/search'

    platform = str(course.get('platform', '')).strip().lower()
    if platform == 'edx':
        return f'https://www.edx.org/search?q={quote_plus(title)}'
    if platform == 'skillshare':
        return f'https://www.skillshare.com/search?query={quote_plus(title)}'
    if platform == 'udemy':
        return f'https://www.udemy.com/courses/search/?q={quote_plus(title)}'
    if platform == 'coursera':
        return f'https://www.coursera.org/search?query={quote_plus(title)}'

    return f'https://www.google.com/search?q={quote_plus(title)}'


def _render_course_buttons(path: List[str], course_index: dict) -> None:
    courses = [course_index.get(name) for name in path]
    courses = [course for course in courses if course]
    if not courses:
        return

    links = []
    for course in courses:
        label = str(course.get('name') or course.get('nombre') or 'Abrir curso')
        link = _course_link(course)
        links.append(
            f'<a class="course-link-word" href="{escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(label)}</a>'
        )

    st.markdown(f'<div class="course-links">{"".join(links)}</div>', unsafe_allow_html=True)


def main():
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

    st.markdown(
        """
        <div class="hero-header">
            <h1>Career Path Explorer</h1>
            <p>Generate, validate, and compare alternative career paths under optimization criteria.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    courses = load_data()
    course_names = [c.get('name') or c.get('nombre') for c in courses]
    course_index = { (c.get('name') or c.get('nombre')): c for c in courses }

    if not course_names:
        st.error('Could not load courses from local data or Kaggle. Check the data/ folder and Kaggle access if needed.')
        return

    default_python_skill = next((name for name in course_names if 'python basics' in name.lower()), None)
    default_skills = [default_python_skill] if default_python_skill else [course_names[0]]
    criterion_options = {
        'Fastest path': 'rapida',
        'Cheapest path': 'economica',
        'Balanced path': 'balanceada',
    }

    with st.form('input_form'):
        user_text = st.text_area('Describe your goal and constraints (for example: "I want to learn data science while avoiding math")', height=120)
        initial_skills = st.multiselect('Skills you already have', options=course_names, default=default_skills)
        selected_criterion = st.radio(
            'Optimization criteria',
            options=list(criterion_options.keys()),
            index=0,
            horizontal=True,
            help='Each criterion produces a different path to compare time, cost, and difficulty.'
        )
        submitted = st.form_submit_button('Generate paths')

    if submitted:
        # If the user submits an empty goal, show a simple message and stop.
        if not (user_text and user_text.strip()):
            st.info('No goal entered.')
            return

        parsed = parse_input(user_text or '')
        # Use the parsed goal if available; otherwise fall back to the raw user text.
        goal = parsed.get('goal') or (user_text.strip() if user_text and user_text.strip() else 'Data Scientist')

        avoid_math = parsed.get('preferences', {}).get('avoid_math', False)
        avoid_cats = {'Matemáticas'} if avoid_math else None

        selected_criteria = [criterion_options[selected_criterion]] if selected_criterion in criterion_options else ['rapida']

        with st.spinner('Generating paths...'):
            paths = generate_paths(
                courses,
                initial_skills,
                goal,
                max_paths=max(1, len(selected_criteria)),
                avoid_categories=avoid_cats,
                user_prefs=parsed.get('preferences', {}),
                criteria_names=selected_criteria,
            )

        st.caption(f'Detected goal: {goal or "not detected"}')
        if parsed.get('skills'):
            st.caption(f'Skills detected by the parser: {", ".join(parsed["skills"])}')

        if not paths:
            st.warning('No valid paths were found. Adjust the goal, constraints, or existing skills.')
        else:
            # Each path will be shown as a subheader (same size as previous 'Generated paths' header)
            # Get brief explanations for each path to show below each entry
            brief_explanations = []
            try:
                brief_explanations = _llm.explain_paths_brief(paths)
            except Exception:
                brief_explanations = []

            for i, p in enumerate(paths, 1):
                target = p.get('target_course') or goal
                criterion = p.get('criterion') or f'Path {i}'
                st.subheader(f"{criterion}: {' → '.join(p['path'])}")
                # show the target course text directly in gray (no label)
                if target:
                    st.caption(target)
                m = p['metrics']
                st.caption(f"Total time: {m['total_months']} months | Cost: ${m['total_cost']} | Average difficulty: {m['avg_difficulty']}")

                # show brief multi-line explanation in gray
                if i - 1 < len(brief_explanations):
                    # st.caption accepts multi-line strings; they will appear in gray
                    st.caption(brief_explanations[i - 1])
                _render_course_buttons(p['path'], course_index)

            st.subheader('Qualitative comparison (LLM)')
            explanation = explain_comparison(paths, {'goal': goal, 'skills': initial_skills, 'objective': goal})
            st.text_area('Explanation', value=explanation, height=240)


if __name__ == '__main__':
    main()
