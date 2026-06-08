"""Streamlit web UI glue for the Career Path Explorer app.

This module wires the local planner and LLM adapter into a lightweight
Streamlit interface. It handles dynamic local imports, form inputs, and
renders course links and generated learning paths.
"""

import streamlit as st      # Web UI framework for building interactive apps
from typing import List     # Type hint for lists
from html import escape     # Escapes HTML to prevent injection
from urllib.parse import quote_plus   # URL‑encodes strings for safe query parameters

# Local planner functions & Local LLM adapter functions
from planner import index_courses, load_courses, generate_paths, build_skill_index  
from llm_adapter import parse_input, explain_comparison, explain_paths_brief  

# A predefined list of skills to populate the multi-select input.
PREDEFINED_SKILLS = [
    # Programación General
    'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'go', 'rust', 'php', 'ruby',
    'kotlin', 'swift', 'r', 'matlab', 'scala', 'haskell', 'perl', 'lua', 'bash scripting',
    'shell scripting', 'powershell', 'groovy', 'dart', 'elixir', 'erlang', 'clojure',
    
    # Frontend Development
    'react', 'vue.js', 'angular', 'svelte', 'next.js', 'nuxt.js', 'gatsby', 'ember.js',
    'html5', 'css3', 'sass', 'less', 'tailwind css', 'bootstrap', 'material design',
    'web components', 'jquery', 'd3.js', 'three.js', 'babylon.js', 'webgl',
    'vanilla javascript', 'webpack', 'vite', 'parcel', 'rollup', 'gulp', 'grunt',
    
    # Backend Development
    'node.js', 'express', 'django', 'flask', 'fastapi', 'spring boot', 'spring mvc',
    'asp.net', 'asp.net core', 'laravel', 'symfony', 'rails', 'sinatra', 'gin',
    'echo', 'fasthttp', 'aiohttp', 'tornado', 'bottle', 'falcon', 'pyramid',
    
    # Bases de Datos
    'sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'redis', 'elasticsearch',
    'cassandra', 'dynamodb', 'oracle', 'sqlserver', 'mariadb', 'cockroachdb',
    'neo4j', 'influxdb', 'timescaledb', 'couchdb', 'arangodb', 'datastore',
    
    # DevOps & Cloud
    'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions',
    'aws', 'azure', 'google cloud', 'heroku', 'netlify', 'vercel', 'cloudflare',
    'ci/cd', 'containerization', 'orchestration', 'infrastructure as code', 'helm',
    'vagrant', 'prometheus', 'grafana', 'elk stack', 'splunk', 'datadog', 'newrelic',
    
    # Testing & QA
    'unit testing', 'integration testing', 'e2e testing', 'test automation', 'jest',
    'mocha', 'chai', 'pytest', 'unittest', 'selenium', 'cypress', 'playwright',
    'appium', 'jmeter', 'postman', 'insomnia', 'soap ui', 'loadrunner',
    
    # Diseño Web
    'figma', 'adobe xd', 'sketch', 'protopie', 'framer', 'webflow', 'ui design',
    'ux design', 'responsive design', 'wireframing', 'prototyping', 'user research',
    'usability testing', 'accessibility', 'wcag', 'design systems', 'color theory',
    'typography', 'layout design', 'information architecture', 'user journey mapping',
    
    # Diseño Digital & Gráfico
    'adobe creative suite', 'photoshop', 'illustrator', 'indesign', 'premiere pro',
    'after effects', 'lightroom', 'xd', 'affinity designer', 'corel draw', 'inkscape',
    'blender', 'cinema 4d', '3d modeling', 'animation', 'motion graphics', 'video editing',
    'color grading', 'graphic design', 'brand identity', 'logo design', 'packaging design',
    
    # Marketing Digital
    'seo', 'sem', 'google ads', 'facebook ads', 'instagram ads', 'linkedin ads',
    'content marketing', 'email marketing', 'affiliate marketing', 'growth hacking',
    'conversion rate optimization', 'a/b testing', 'analytics', 'google analytics',
    'mixpanel', 'amplitude', 'segment', 'social media marketing', 'influencer marketing',
    'community management', 'pr', 'copywriting', 'brand strategy',
    
    # Mobile Development
    'react native', 'flutter', 'ionic', 'xamarin', 'swiftui', 'jetpack compose',
    'android development', 'ios development', 'mobile design', 'cross-platform development',
    'progressive web apps', 'pwa', 'offline-first', 'mobile ux',
    
    # Data Science & Analytics
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
    'pandas', 'numpy', 'data analysis', 'statistical analysis', 'data visualization',
    'tableau', 'power bi', 'looker', 'qlik', 'apache spark', 'hadoop', 'hive',
    'nlp', 'computer vision', 'predictive modeling', 'time series analysis',
    'feature engineering', 'data wrangling', 'jupyter notebooks',
    
    # Arquitectura & Patrones
    'microservices', 'monolith', 'rest api', 'graphql', 'websockets', 'grpc',
    'message queues', 'event-driven architecture', 'cqrs', 'saga pattern',
    'design patterns', 'solid principles', 'clean code', 'refactoring',
    'scalability', 'performance optimization', 'caching strategies',
    
    # Seguridad
    'cybersecurity', 'penetration testing', 'web security', 'owasp top 10',
    'encryption', 'ssl/tls', 'oauth2', 'jwt', 'authentication', 'authorization',
    'api security', 'secrets management', 'vulnerability assessment', 'security testing',
    'ethical hacking', 'compliance', 'gdpr', 'pci dss', 'hipaa',
    
    # Soft Skills
    'communication', 'problem solving', 'critical thinking', 'time management',
    'project management', 'agile', 'scrum', 'kanban', 'leadership', 'teamwork',
    'collaboration', 'adaptability', 'learning ability', 'creativity', 'innovation',
    'public speaking', 'presentation skills', 'documentation', 'mentoring',
    
    # Tools & Platforms
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'notion',
    'asana', 'monday.com', 'trello', 'linear', 'visual studio code', 'intellij idea',
    'xcode', 'android studio', 'pycharm', 'jupyter', 'colab', 'vercel', 'netlify',
    
    # Otros Lenguajes & Tecnologías
    'graphql', 'apollo', 'prisma', 'sequelize', 'typeorm', 'sqlalchemy',
    'message brokers', 'rabbitmq', 'kafka', 'mqtt', 'nats', 'aws lambda',
    'azure functions', 'google cloud functions', 'serverless', 'edge computing',
    'blockchain', 'web3', 'solidity', 'ethereum', 'smart contracts',
]


# ===========================================================================
# AUXILIARY METHODS
# ===========================================================================

def _merge_skill_lists(*skill_groups: List[str]) -> List[str]:
    """Merge multiple skill groups into a single ordered list while removing duplicates.

    The function preserves the original order of the first occurrence of each skill and
    deduplicates case-insensitively.
    """
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
    """Return a best-effort URL for a course record.

    Prefer explicit link fields when present, fall back to a platform-specific
    search page for known providers, and otherwise use a Google search.
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
    """Render styled HTML links for each course in the provided path."""
    
    courses = [course_index.get(name) for name in path]
    courses = [course for course in courses if course]

    if not courses:
        return

    links = []

    for course in courses:
        label = str(
            course.get("name")
            or course.get("nombre")
            or "Open course"
        )

        link = _course_link(course)

        links.append(
            f'<a href="{escape(link, quote=True)}" '
            f'target="_blank" '
            f'rel="noopener noreferrer" '
            f'style="'
            f'display:inline-block;'
            f'padding:4px 10px;'
            f'background:rgba(15,23,42,0.75);'
            f'border:1px solid rgba(255,255,255,0.05);'
            f'border-radius:12px;'
            f'box-shadow:0 4px 12px rgba(0,0,0,0.25);'
            f'color:#4da3ff;'
            f'text-decoration:none;'
            f'font-size:0.95rem;'
            f'font-weight:500;'
            f'white-space:nowrap;'
            f'">'
            f'{escape(label)}'
            f'</a>'
        )

    html = (
        '<div style="'
        'display:flex;'
        'flex-wrap:wrap;'
        'gap:8px;'
        'margin-top:1rem;'
        'margin-bottom:1rem;'
        '">'
        + "".join(links)
        + "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)

# ===========================================================================
# APP ORCHESTRATOR (MAIN)
# ===========================================================================

def main():
    """Run the Streamlit app and coordinate UI, input, and path generation.

    This function loads course data, collects user goals and skills, generates
    candidate paths, and displays metrics plus optional LLM explanations.
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
    # Build a skill index to populate the 'skills you already have' selector.
    skill_index = build_skill_index(courses)

    # Choose sensible defaults that actually exist in the options.
    default_skills = 'python'
    # Build a form to capture the user's goal, existing skills, and optimization criterion.
    with st.form('input_form'):
        user_text = st.text_area('Describe your goal and constraints (eg: "I want to learn data science while avoiding math")', height=120)
        initial_skills = st.multiselect('Skills you already have', options=sorted(PREDEFINED_SKILLS), default=default_skills)
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

        # Show the loading spinner and keep it active until ALL processing is complete.
        # This includes: parsing, path generation, LLM explanations, and validation.
        with st.spinner('Generating paths...'):
            # Parse free-form user text into a structured profile (goal, preferences, skills, avoids).
            goal, preferences, parser_skills, parser_avoids = parse_input(user_text or '')
            user_skills = _merge_skill_lists(initial_skills, parser_skills)

            # Generate candidate paths using the planner. This may be CPU-bound.
            paths = generate_paths(
                courses,
                user_skills,
                goal,
                max_paths=5,
                avoid_categories=parser_avoids,
                user_prefs=preferences,
                criterion_name=selected_criterion,
            )

            # Prepare all data for rendering, including LLM explanations.
            # Do not show any UI output until everything is ready.
            brief_explanations = []
            full_comparison_explanation = ''
            
            # Only attempt to generate explanations if paths were found.
            if paths:
                # Get brief explanations for each path from the LLM helper (optional).
                try:
                    brief_explanations = explain_paths_brief(paths, goal)
                except Exception:
                    # If the LLM helper fails, the UI can still render the raw paths.
                    brief_explanations = []

                # Generate qualitative comparison from the LLM helper.
                try:
                    full_comparison_explanation = explain_comparison(
                        paths,
                        {
                            'goal': goal,
                            'skills': user_skills,
                            'preferences': preferences,
                            'avoids': parser_avoids,
                        },
                    )
                except Exception:
                    full_comparison_explanation = ''
            
            # Store all processed data in session state to render without additional processing.
            st.session_state.generation_complete = True
            st.session_state.goal = goal
            st.session_state.paths = paths
            st.session_state.brief_explanations = brief_explanations
            st.session_state.full_comparison_explanation = full_comparison_explanation
            st.session_state.user_skills = user_skills

        # After the spinner closes, render the completed results from session state.
        # This ensures no partial results are shown and no additional processing occurs.
        if st.session_state.get('generation_complete', False):
            goal = st.session_state.goal
            paths = st.session_state.paths
            brief_explanations = st.session_state.brief_explanations
            full_comparison_explanation = st.session_state.full_comparison_explanation

            # Show detected goal and skills to the user for transparency.
            st.caption(f'Detected goal: {goal or "not detected"}')

            if not paths:
                st.warning('No valid paths were found. Adjust the goal, constraints, or existing skills.')
            else:
                # Render each path with metrics, explanation, and course links.
                for i, p in enumerate(paths, 1):
                    target = p.get('target_course') or goal
                    criterion = f'Path {i}'
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
                st.text_area('Explanation', value=full_comparison_explanation, height=240)

if __name__ == '__main__':
    main()
