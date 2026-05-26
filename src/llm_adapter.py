import json
import os
import urllib.error
import urllib.request
from typing import Dict, List

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN') or os.environ.get('HF_TOKEN')
HF_MODEL = os.environ.get('HUGGINGFACE_MODEL', 'google/flan-t5-small')

if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
    except Exception:
        openai = None
else:
    openai = None


COMMON_ROLES = ["AI Engineer", "Data Scientist", "Cloud Engineer", "Backend Engineer", "Cybersecurity Analyst"]


def parse_input(text: str) -> Dict:
    # If openai available, use it to parse (best-effort). Otherwise fallback to simple heuristics.
    if openai:
        try:
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': f"Parse this into JSON goal/preferences/skills: {text}"}],
                temperature=0
            )
            content = resp['choices'][0]['message']['content']
            return json.loads(content)
        except Exception:
            pass

    if HF_TOKEN:
        try:
            prompt = (
                'Convierte el siguiente texto en un JSON estricto con las claves '
                'goal, preferences y skills. preferences debe ser un objeto. '
                f'Texto: {text}'
            )
            payload = json.dumps({'inputs': prompt}).encode('utf-8')
            request = urllib.request.Request(
                f'https://api-inference.huggingface.co/models/{HF_MODEL}',
                data=payload,
                headers={
                    'Authorization': f'Bearer {HF_TOKEN}',
                    'Content-Type': 'application/json'
                },
                method='POST'
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read().decode('utf-8')
            data = json.loads(raw)
            if isinstance(data, list) and data:
                generated = data[0].get('generated_text', '')
            else:
                generated = data.get('generated_text', '') if isinstance(data, dict) else ''
            if generated:
                start = generated.find('{')
                end = generated.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return json.loads(generated[start:end + 1])
        except Exception:
            pass

    # Heuristic fallback
    lower = text.lower()
    goal = None
    for r in COMMON_ROLES:
        if r.lower() in lower:
            goal = r
            break
    if not goal:
        # pick first known role word
        for r in COMMON_ROLES:
            if r.split()[0].lower() in lower:
                goal = r
                break
    preferences = {}
    preferences['avoid_math'] = 'matem' in lower or 'math' in lower

    # naive skills extraction: find mentions of some keywords
    skills = []
    for kw in ['python', 'sql', 'docker', 'kubernetes', 'aws', 'statistics', 'linear algebra', 'machine learning']:
        if kw in lower:
            skills.append(kw.capitalize() if ' ' not in kw else kw.title())

    return {
        'goal': goal or '',
        'preferences': preferences,
        'skills': skills
    }


def explain_comparison(paths: List[Dict], user_profile: Dict) -> str:
    # If OpenAI available, ask for qualitative comparison
    if openai:
        try:
            prompt = 'Compare these career paths and explain advantages and trade-offs for a user: '\
                     + str(paths) + '\nUser: ' + str(user_profile)
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return resp['choices'][0]['message']['content']
        except Exception:
            pass

    if HF_TOKEN:
        try:
            summary = []
            for i, p in enumerate(paths, 1):
                m = p.get('metrics', {})
                summary.append(
                    f'Ruta {i}: {" -> ".join(p.get("path", []))}. '
                    f'Tiempo {m.get("total_months")} meses, coste {m.get("total_cost")}, '
                    f'dificultad media {m.get("avg_difficulty")}'
                )
            prompt = (
                'Compara estas trayectorias profesionales de forma clara, breve y útil. '
                'Destaca rapidez, esfuerzo y conveniencia para el usuario.\n'
                + '\n'.join(summary)
                + f'\nPerfil del usuario: {user_profile}'
            )
            payload = json.dumps({'inputs': prompt}).encode('utf-8')
            request = urllib.request.Request(
                f'https://api-inference.huggingface.co/models/{HF_MODEL}',
                data=payload,
                headers={
                    'Authorization': f'Bearer {HF_TOKEN}',
                    'Content-Type': 'application/json'
                },
                method='POST'
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read().decode('utf-8')
            data = json.loads(raw)
            if isinstance(data, list) and data:
                generated = data[0].get('generated_text', '')
            else:
                generated = data.get('generated_text', '') if isinstance(data, dict) else ''
            if generated:
                return generated
        except Exception:
            pass

    # Simple fallback explanation
    lines = []
    for i, p in enumerate(paths, 1):
        m = p.get('metrics', {})
        lines.append(f"Ruta {i}: {', '.join(p.get('path', []))}\n - Tiempo: {m.get('total_months')} meses, Coste aprox: ${m.get('total_cost')}, Dificultad media: {m.get('avg_difficulty')}")
    lines.append('\nSugerencia: elige la ruta con menor tiempo si buscas inserción rápida; elige mayor dificultad si buscas profundidad técnica.')
    return '\n\n'.join(lines)
