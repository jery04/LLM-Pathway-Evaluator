import streamlit as st
from typing import List

try:
    from planner import load_courses, generate_paths
    from llm_adapter import parse_input, explain_comparison
except ModuleNotFoundError:
    from src.planner import load_courses, generate_paths
    from src.llm_adapter import parse_input, explain_comparison


@st.cache_data
def load_data():
    return load_courses()


def main():
    st.set_page_config(page_title='Career Path Explorer AI')

    st.title('Career Path Explorer AI')
    st.markdown('Genera y compara trayectorias profesionales alternativas.')

    courses = load_data()
    course_names = [c['nombre'] for c in courses]

    if not course_names:
        st.error('No se pudieron cargar cursos desde Kaggle. Revisa que tengas kagglehub instalado y acceso al dataset.')
        return

    default_skills = ['Python'] if 'Python' in course_names else [course_names[0]]

    with st.form('input_form'):
        user_text = st.text_area('Describe tu objetivo y preferencias (ej: "Quiero trabajar en IA sin demasiada matemática")', height=120)
        initial_skills = st.multiselect('Habilidades que ya tienes', options=course_names, default=default_skills)
        weekly_time = st.number_input('Horas semanales disponibles', min_value=1, max_value=168, value=8)
        submitted = st.form_submit_button('Generar rutas')

    if submitted:
        parsed = parse_input(user_text or '')
        goal = parsed.get('goal') or 'Data Scientist'

        avoid_math = parsed.get('preferences', {}).get('avoid_math', False)
        avoid_cats = {'Matemáticas'} if avoid_math else None

        st.info('Generando rutas...')
        paths = generate_paths(courses, initial_skills, goal, max_paths=3, avoid_categories=avoid_cats)

        st.caption(f'Objetivo detectado: {goal or "no detectado"}')
        if parsed.get('skills'):
            st.caption(f'Habilidades detectadas por el parser: {", ".join(parsed["skills"])}')

        if not paths:
            st.warning('No se encontraron rutas. Ajusta objetivo o habilidades iniciales.')
        else:
            st.subheader('Rutas generadas')
            for i, p in enumerate(paths, 1):
                st.markdown(f"**Ruta {i}**: {' → '.join(p['path'])}")
                m = p['metrics']
                st.write(f"Tiempo total: {m['total_months']} meses | Coste: ${m['total_cost']} | Dificultad media: {m['avg_difficulty']}")

            st.subheader('Comparación cualitativa (LLM)')
            explanation = explain_comparison(paths, {'goal': goal, 'skills': initial_skills, 'weekly_time': weekly_time})
            st.text_area('Explicación', value=explanation, height=240)


if __name__ == '__main__':
    main()
