"""Streamlit web UI glue for the Career Path Explorer.

This module wires the local planner and LLM adapter into a lightweight
Streamlit interface. It handles dynamic imports, form inputs, course link
rendering, and parallel LLM-based path explanations.
"""

import re                              # regex for text pattern matching
import streamlit as st                 # web app framework
from typing import List                # type hint for lists
from html import escape                # sanitize HTML output
from urllib.parse import quote_plus    # URL encode strings
from concurrent.futures import ThreadPoolExecutor  # parallel LLM calls
import simulation as sim

# Load data, generate paths, and call LLM for explanations
from planner import index_courses, load_courses, generate_paths, build_skill_index, _load_embeddings_data  
from llm_adapter import parse_input, explain_comparison, explain_paths_brief  

# Predefined skill list for user selection
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

# Pre-sorted once at import time — never re-sorted on each render
PREDEFINED_SKILLS_SORTED = sorted(PREDEFINED_SKILLS)

# CSS and hero HTML as module-level constants
_APP_STYLES = """
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
.comparison-card {
    background: linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(10,14,24,0.98) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 28px 80px rgba(0,0,0,0.35);
    border-radius: 22px;
    padding: 1.5rem 1.6rem;
    margin: 1.5rem 0 2rem;
    color: rgba(235,235,245,0.95);
}
.comparison-card h2 {
    margin: 0 0 0.75rem;
    font-size: clamp(1.6rem, 2.4vw, 2rem);
    letter-spacing: -0.03em;
    color: #ffffff;
}
.comparison-card p {
    margin: 0.85rem 0;
    line-height: 1.72;
    color: rgba(220,220,230,0.92);
    font-size: 1rem;
}
.comparison-card p:first-child { margin-top: 0; }
.comparison-card p:last-child  { margin-bottom: 0; }
.comparison-card strong { color: #8ab8ff; }
</style>
"""

_HERO_HTML = """
<div class="hero-header">
    <h1>Career Path Explorer 🚀</h1>
    <p>Generate, validate, and compare alternative career paths under optimization criteria.</p>
</div>
"""


# ===========================================================================
# AUXILIARY METHODS
# ===========================================================================

@st.cache_data
def _cached_load():
    """Load and index the course catalog exactly once per Streamlit session."""
    courses = load_courses()
    course_index = index_courses(courses)
    skill_index = build_skill_index(courses)
    embeddings_data = _load_embeddings_data(experiment=False)
    return courses, course_index, skill_index, embeddings_data

def _merge_skill_lists(*skill_groups: List[str]) -> List[str]:
    """Merge multiple skill groups into a single ordered list while removing duplicates."""
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
    """Return a best-effort URL for a course record."""
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
    """Streamlit app entry point.

    Flow:
        1. Setup: Configure page, load courses (cached).
        2. On submit: Parse user input, generate paths, run LLM explanations in parallel.
        3. On success: Display paths, metrics, and LLM comparison.
    """
    st.set_page_config(page_title='Career Path Explorer')
    st.markdown(_APP_STYLES, unsafe_allow_html=True)
    st.markdown(_HERO_HTML, unsafe_allow_html=True)

    # Load course catalog once — subsequent reruns hit the cache instantly
    courses, course_index, skill_index, embeddings_data = _cached_load()

    default_skills = 'python'
    with st.form('input_form'):
        user_text = st.text_area(
            'Describe your goal',
            height=120,
            placeholder='I want to learn web design...'
        )
        initial_skills = st.multiselect(
            'Skills you already have',
            options=PREDEFINED_SKILLS_SORTED,
            default=default_skills,
        )
        selected_criterion = st.radio(
            'Optimization criteria',
            options=['Fastest path', 'Cheapest path', 'Balanced path'],
            index=0,
            horizontal=True,
            help='Each criterion produces a different path to compare time, cost, and difficulty.'
        )
        submitted = st.form_submit_button('Generate paths')

    if submitted:
        if not (user_text and user_text.strip()):
            st.info('No goal entered.')
            return

        with st.spinner('Generating paths. This may take a few minutes...'):
            goal, preferences, parser_skills, parser_avoids = parse_input(user_text or '')
            user_skills = _merge_skill_lists(initial_skills, parser_skills)

            paths = generate_paths(
                courses,
                user_skills,
                goal,
                max_paths=5,
                avoid_categories=parser_avoids,
                user_prefs=preferences,
                criterion_name=selected_criterion,
                course_index=course_index,
                skill_index=skill_index,
                embeddings_data=embeddings_data,
            )

            brief_explanations = []
            full_comparison_explanation = ''

            if paths:
                context = {
                    'goal': goal,
                    'skills': user_skills,
                    'preferences': preferences,
                    'avoids': parser_avoids,
                }

                # Run both LLM calls in parallel — saves the full serial wait time
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_brief = executor.submit(explain_paths_brief, paths, goal)
                    future_full  = executor.submit(explain_comparison, paths, context)

                    try:
                        brief_explanations = future_brief.result()
                    except Exception:
                        brief_explanations = []

                    try:
                        full_comparison_explanation = future_full.result()
                    except Exception:
                        full_comparison_explanation = ''

            st.session_state.generation_complete = True
            st.session_state.goal = goal
            st.session_state.paths = paths
            st.session_state.brief_explanations = brief_explanations
            st.session_state.full_comparison_explanation = full_comparison_explanation
            st.session_state.user_skills = user_skills

    if st.session_state.get('generation_complete', False):
        goal = st.session_state.goal
        paths = st.session_state.paths
        brief_explanations = st.session_state.brief_explanations
        full_comparison_explanation = st.session_state.full_comparison_explanation

        st.caption(f'Detected goal: {goal or "not detected"}')

        if not paths:
            st.warning('No valid paths were found. Adjust the goal, constraints, or existing skills.')
        else:
            for i, p in enumerate(paths, 1):
                target = p.get('target_course') or goal
                criterion = f'Path {i}'
                st.subheader(f"{criterion}: {' → '.join(p['path'])}")
                if target:
                    st.caption(target)
                m = p['metrics']
                st.caption(f"Total time: {m['total_months']} months | Cost: ${m['total_cost']} | Average difficulty: {m['avg_difficulty']}")

                if i - 1 < len(brief_explanations):
                    st.caption(brief_explanations[i - 1])
                _render_course_buttons(p['path'], course_index)

            comparison_text = full_comparison_explanation or 'La comparación se generará aquí una vez que termine el análisis.'
            comparison_text = escape(comparison_text)
            comparison_text = re.sub(
                r'\*\*(.+?)\*\*',
                r'<span style="color:#8ab8ff;font-weight:700;">\1</span>',
                comparison_text,
            )
            comparison_text = comparison_text.replace('\n', '<br>')
            st.markdown(
                f"""
                <div class="comparison-card">
                    <h2>Qualitative comparison (LLM)</h2>
                    <div style="font-size: 1rem; color: rgba(230,230,240,0.95); line-height:1.75;">
                        {comparison_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -------------------------
            # Simulation section (interactive viewer)
            # -------------------------
            st.header('Simulation')
            st.caption('Monte Carlo simulation — statistically estimates completion rates, time ranges, and dropout risks for each learning path.')

            with st.expander('Simulation controls'):
                n_simulations = st.slider('Number of simulations', 100, 5000, 500, step=100)
                hours_per_week = st.slider('Hours of study per week', 5, 40, 15)
                profile = st.selectbox('Profile de estudiante', ['Principiante', 'Intermedio', 'Avanzado', 'Trabajador'])
                fatigue_tolerance = st.slider('Tolerancia a la fatiga', 0.05, 0.5, 0.15, step=0.01)
                dropout_tolerance = st.slider('Dropout tolerance', 0.05, 0.5, 0.2, step=0.01)
                run_sim = st.button('Run simulation')

            if run_sim:
                paths_to_sim = st.session_state.get('paths') or []

                # Map UI controls to LearningParameters
                if profile == 'Principiante':
                    mu, sigma, weekly_scale = 2.2, 0.6, 0.8
                elif profile == 'Intermedio':
                    mu, sigma, weekly_scale = 2.0, 0.5, 1.0
                elif profile == 'Avanzado':
                    mu, sigma, weekly_scale = 1.7, 0.4, 1.2
                else:  # Trabajador
                    mu, sigma, weekly_scale = 1.9, 0.45, 1.5

                params = sim.LearningParameters(
                    learning_rate_mu=mu,
                    learning_rate_sigma=sigma,
                    weekly_capacity_lambda=max(1, int(hours_per_week * weekly_scale)),
                    fatigue_multiplier=float(fatigue_tolerance),
                    dropout_scale=float(20 * (1 + dropout_tolerance * 4)),
                )

                with st.spinner('Running simulations...'):
                    ui_data = sim.get_simulation_ui_data(course_index, paths_to_sim, params, n_simulations)

                results = ui_data.get('results', [])
                if not results:
                    st.warning('No simulation results were produced.')
                else:
                    df = sim.create_comparison_table(results)
                    # Ensure df is shown in path_index order for clarity
                    if 'path_index' in df.columns:
                        df = df.sort_values('path_index')
                    st.subheader('Comparison between paths')
                    st.dataframe(df)

                    # Completion rate bar chart
                    st.subheader('Completion rate by path')
                    comp_series = df.set_index('path_name')['completion_rate']
                    st.bar_chart(comp_series)

                    # Aggregate across all paths: combine completion times and show overall statistics
                    combined_times = []
                    for r in results:
                        try:
                            t = getattr(r, 'completion_times')
                        except Exception:
                            try:
                                t = r.get('completion_times', [])
                            except Exception:
                                t = []
                        if t:
                            combined_times.extend(list(t))

                    if combined_times:
                        st.subheader('Completion time distribution — All paths')
                        try:
                            fig = sim.plot_time_distribution(combined_times, 'All paths')
                            st.pyplot(fig)
                        except Exception:
                            pass

                        try:
                            fig_box = sim.plot_time_boxplot(combined_times, 'All paths')
                            st.pyplot(fig_box)
                        except Exception:
                            pass

                        try:
                            fig_cdf = sim.plot_time_cdf(combined_times, 'All paths')
                            st.pyplot(fig_cdf)
                        except Exception:
                            pass
                    else:
                        st.info('No completion time data available to show aggregated distributions.')

                    # Aggregate course failure statistics and show useful summaries
                    bottlenecks = ui_data.get('course_bottlenecks', {})
                    if bottlenecks:
                        import pandas as _pd
                        # `bottlenecks` stores estimated failure probability per course
                        fail_df = _pd.Series(bottlenecks).rename('failure_rate').to_frame()
                        fail_df['success_rate'] = 1.0 - fail_df['failure_rate']
                        fail_df = fail_df.sort_values('failure_rate', ascending=False)

                        st.subheader('Top courses by estimated failure rate')
                        # Show top 10 courses most likely to cause failures
                        st.table(fail_df.head(10).style.format({ 'failure_rate': '{:.1%}', 'success_rate': '{:.1%}' }))

                        st.subheader('Failure / Success rates (all courses)')
                        # Provide a compact bar chart of failure rates for the worst courses
                        st.bar_chart(fail_df['failure_rate'].head(20))

                        # Also expose a downloadable CSV for further inspection
                        try:
                            csv_bytes = fail_df.reset_index().to_csv(index=False).encode('utf-8')
                            st.download_button('Download course failure rates (CSV)', data=csv_bytes, file_name='course_failure_rates.csv', mime='text/csv')
                        except Exception:
                            pass

                    # Aggregate fatigue risk (show median/mean across paths)
                    fatigue_vals = []
                    for r in results:
                        try:
                            fatigue_vals.append(float(getattr(r, 'fatigue_risk', 0.0)))
                        except Exception:
                            try:
                                fatigue_vals.append(float(r.get('fatigue_risk', 0.0)))
                            except Exception:
                                pass

                    if fatigue_vals:
                        import statistics as _st
                        mean_fatigue = _st.mean(fatigue_vals)
                        display_progress = min(1.0, max(0.01, mean_fatigue))
                        st.subheader('Fatigue risk (aggregated)')
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric('Mean fatigue risk', f"{mean_fatigue:.2%}")
                        with col2:
                            st.progress(display_progress)

                    # Additional aggregated stats
                    st.subheader('Additional aggregated statistics')
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric('Avg completion', f"{ui_data['summary_stats']['avg_completion_rate']:.1%}")
                    with stats_col2:
                        st.metric('Avg time (weeks)', f"{ui_data['summary_stats']['avg_time_weeks']:.1f}")
                    with stats_col3:
                        st.metric('Avg dropout', f"{ui_data['summary_stats']['avg_dropout_prob']:.1%}")
                    
                    # Dropout distribution across paths
                    try:
                        dropouts = [r.get('dropout_prob', None) if isinstance(r, dict) else getattr(r, 'dropout_prob', None) for r in results]
                        dropouts = [d for d in dropouts if d is not None]
                        if dropouts:
                            st.subheader('Dropout probability distribution (per simulation result)')
                            import numpy as _np
                            hist_vals = _np.histogram(dropouts, bins=10)
                            # show a simple dataframe for bins
                            bins = [f"{hist_vals[1][i]:.2f}-{hist_vals[1][i+1]:.2f}" for i in range(len(hist_vals[1])-1)]
                            bin_df = _pd.DataFrame({'bin': bins, 'count': list(hist_vals[0])})
                            st.table(bin_df)
                            st.bar_chart(_pd.Series(dropouts).rename('dropout_prob'))
                    except Exception:
                        pass

if __name__ == '__main__':
    main()
