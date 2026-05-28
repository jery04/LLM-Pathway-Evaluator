import json
import re
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
                'Convert the following text into a strict JSON with keys '
                'goal, preferences and skills. preferences must be an object. '
                f'Text: {text}'
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
    preferences = {}

    # detect explicit mentions of wanting/learning (English/Spanish)
    m = re.search(r"(?:i want to learn to|i want to learn|i want to study|quiero aprender a|quiero aprender|aprender)(?:\s+about|\s+to|\s+)([a-záéíóúñ0-9 ,]+)", lower)
    if m:
        raw_goal = m.group(1).strip().strip('.!?,')
        # map some Spanish/common phrases to English course/goal names
        mapping = {
            'estadística computacional': 'Computational Statistics',
            'estadistica computacional': 'Computational Statistics',
            'estadística': 'Statistics',
            'estadistica': 'Statistics',
            'estadistica computacional': 'Computational Statistics',
            'estadistica aplicada': 'Applied Statistics',
            'estadistica aplicada': 'Applied Statistics',
        }
        goal = mapping.get(raw_goal, raw_goal.title())

    # fallback: look for role-like keywords
    if not goal:
        for r in COMMON_ROLES:
            if r.lower() in lower:
                goal = r
                break

    # preferences: detect if user says they don't know Python (english/spanish)
    lacks_python_patterns = [r"don't know python", r"dont know python", r"do not know python", r"no sé python", r"no se python", r"no conozco python", r"no conozco a python", r"no tengo experiencia en python"]
    knows_python = True
    for pat in lacks_python_patterns:
        if pat in lower:
            knows_python = False
            break
    # if user mentions Python positively, set knows_python True
    if 'python' in lower and knows_python:
        knows_python = True

    preferences['knows_python'] = knows_python
    preferences['avoid_math'] = 'matem' in lower or 'math' in lower

    # extract skills mentioned (simple heuristics English/Spanish)
    skills = []
    # explicit mentions like 'I know basic statistics' / 'conozco estadística básica'
    if re.search(r"(basic statistics|estad[ií]stica basica|estad[ií]stica b[aá]sica|basic stats)", lower):
        skills.append('Basic Statistics')

    if 'python' in lower and knows_python:
        skills.append('Python')

    # common skill keywords
    for kw in ['sql', 'docker', 'kubernetes', 'aws', 'statistics', 'linear algebra', 'machine learning']:
        if kw in lower and kw not in ('statistics',) :
            skills.append(kw.capitalize() if ' ' not in kw else kw.title())
    # always include 'Statistics' if user mentions estadistica or statistics
    if 'estad' in lower or 'statistics' in lower:
        if 'Basic Statistics' not in skills:
            skills.append('Statistics')

    return {
        'goal': goal or '',
        'preferences': preferences,
        'skills': skills
    }


def _format_paths_for_comparison(paths: List[Dict]) -> List[str]:
    summaries = []
    for i, path in enumerate(paths, 1):
        metrics = path.get('metrics', {})
        route = ' -> '.join(path.get('path', []))
        target = path.get('target_course') or path.get('objective') or ''
        summaries.append(
            f'Path {i}: {route}. '
            f'Destination: {target or "unspecified"}. '
            f'Time: {metrics.get("total_months", "?")} months. '
            f'Cost: {metrics.get("total_cost", "?")}. '
            f'Avg difficulty: {metrics.get("avg_difficulty", "?")}. '
            f'Steps: {metrics.get("steps", "?")}'
        )
    return summaries


def explain_comparison(paths: List[Dict], user_profile: Dict) -> str:
    # Prefer LLMs if available to produce explanations in the requested format (English).
    prompt_header = (
        'Format the output in English. For each path, show on one line the sequence of courses separated by " -> " and below write a clear explanation of how that path enables reaching the goal. '
        'At the end of all paths, provide an extended and well-argued conclusion indicating which path you recommend and why. '
        'Use a professional, pedagogical tone. Plain text output.'
    )

    summary_lines = []
    for i, p in enumerate(paths, 1):
        route = ' -> '.join(p.get('path', []))
        metrics = p.get('metrics', {})
        summary_lines.append(f'Path {i}: {route}. Time: {metrics.get("total_months", "?")} months; Cost: ${metrics.get("total_cost", "?")}; Avg difficulty: {metrics.get("avg_difficulty", "?")}')

    prompt_body = '\n'.join(summary_lines) + '\nUser profile: ' + str(user_profile)

    if openai:
        try:
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': 'You are an academic advisor who lists learning paths and argues a final recommendation in English.'},
                    {'role': 'user', 'content': prompt_header + '\n\n' + prompt_body}
                ],
                temperature=0.7,
                max_tokens=900
            )
            return resp['choices'][0]['message']['content']
        except Exception:
            pass

    if HF_TOKEN:
        try:
            payload = json.dumps({'inputs': prompt_header + '\n\n' + prompt_body}).encode('utf-8')
            request = urllib.request.Request(
                f'https://api-inference.huggingface.co/models/{HF_MODEL}',
                data=payload,
                headers={
                    'Authorization': f'Bearer {HF_TOKEN}',
                    'Content-Type': 'application/json'
                },
                method='POST'
            )
            with urllib.request.urlopen(request, timeout=60) as response:
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

    # Fallback heuristic: generate explanations and a long recommendation in English.
    def _route_explanation(route_idx: int, p: Dict) -> str:
        path = p.get('path', [])
        metrics = p.get('metrics', {})
        if not path:
            return f'Path {route_idx}: (empty)\nNo courses available.'

        explanation = []
        explanation.append(f"{' -> '.join(path)}")
        # Describe progression: foundations, intermediate, then target.
        if len(path) == 1:
            explanation.append(f"This path consists of a single course that acts as a gateway toward the goal. With an estimated duration of {metrics.get('total_months', '?')} months, it enables acquisition of the essential competencies directly.")
        else:
            first = path[0]
            last = path[-1]
            middle = path[1:-1]
            explanation.append(f"It starts with '{first}', which provides the necessary foundations. It then continues with {', '.join([f"'{m}'" for m in middle])} to consolidate intermediate and practical skills, and finally '{last}' focuses on the final objective.")
            explanation.append(f"Approximate total time: {metrics.get('total_months', '?')} months; estimated cost: ${metrics.get('total_cost', '?')}; average difficulty: {metrics.get('avg_difficulty', '?')}.")
            explanation.append("This sequence respects dependencies: each step builds on the previous one to reduce failures and accelerate practical assimilation.")

        return '\n'.join(explanation)

    parts = []
    for i, p in enumerate(paths, 1):
        parts.append(_route_explanation(i, p))

    # Recommendation heuristics
    def _score(p: Dict) -> float:
        m = p.get('metrics', {})
        months = m.get('total_months') or 1
        cost = m.get('total_cost') or 0
        diff = m.get('avg_difficulty') or 1
        steps = m.get('steps') or max(1, len(p.get('path', [])))
        # Prefer lower months, moderate difficulty, fewer steps, lower cost
        return (1.0 / months) * 0.45 + (1.0 / (1 + diff)) * 0.25 + (1.0 / (1 + steps)) * 0.15 + (1.0 / (1 + cost)) * 0.15

    best_idx = max(range(len(paths)), key=lambda i: _score(paths[i])) if paths else None

    # Long, argued recommendation
    recommendation = []
    recommendation.append('Final conclusion and recommendation:')
    if best_idx is None:
        recommendation.append('It was not possible to determine a recommended path due to insufficient data.')
    else:
        chosen = paths[best_idx]
        rec_route = ' -> '.join(chosen.get('path', []))
        recommendation.append(f'I recommend taking Path {best_idx + 1}: {rec_route}.')
        recommendation.append('Detailed argument:')
        # Provide an extended argument comparing metrics
        for i, p in enumerate(paths, 1):
            m = p.get('metrics', {})
            recommendation.append(
                f'- Path {i}: time {m.get("total_months", "?")} months, cost ${m.get("total_cost", "?")}, avg difficulty {m.get("avg_difficulty", "?")}, steps {m.get("steps", "?")}'
            )

        recommendation.append('I weighed the key factors:')
        recommendation.append('- Speed to reach the goal (I favor paths with lower total time).')
        recommendation.append('- Cognitive load and drop-out risk (I favor moderate difficulty).')
        recommendation.append('- Cost and economic accessibility.')
        recommendation.append('- Technical robustness: I prefer paths that include fundamentals before advanced courses.')

        recommendation.append('In balance, the recommended path offers the best compromise between arriving at the goal quickly without sacrificing the technical depth required to perform competently. If your main priority is minimizing time at all costs, choose a shorter path; if instead you seek maximum technical robustness, choose the longer, deeper path. This recommendation assumes the weekly time reported in the profile and a desire for a balance between speed and robustness.')

        recommendation.append('Practical suggestions:')
        recommendation.append('- Keep a weekly study plan with concrete goals per course.')
        recommendation.append('- Do small projects after each intermediate course to consolidate learning.')
        recommendation.append('- Review and adjust the path as you progress and as market requirements evolve.')

    return '\n\n'.join(parts + ['\n'.join(recommendation)])


def explain_paths_brief(paths: List[Dict]) -> List[str]:
    """Return a short, one-paragraph explanation for each path (English).

    This is used by the UI to show a concise caption under each path.
    """
    brief = []

    for p in paths:
        path = p.get('path', [])
        m = p.get('metrics', {})
        months = m.get('total_months', '?')
        cost = m.get('total_cost', '?')
        difficulty = m.get('avg_difficulty', '?')

        if not path:
            brief.append('No courses available for this path.')
            continue

        # Compose a short 3-line paragraph describing progression, skills gained, and practical advice.
        if len(path) == 1:
            line1 = f"Single course '{path[0]}' serves as a focused entry point to the topic."
            line2 = f"This course introduces the essential concepts and practical skills needed to start working in the area."
            line3 = f"Estimated time: {months} months; cost: ${cost}; average difficulty: {difficulty}."
            s = '\n'.join([line1, line2, line3])
        else:
            first = path[0]
            last = path[-1]
            middle = path[1:-1]
            line1 = f"Starts with '{first}' to build foundations, then {' then '.join([m for m in middle]) if middle else 'continues'} and finishes with '{last}'."
            line2 = f"This sequence builds practical skills step-by-step: foundations → intermediate practice → applied project or specialization."
            line3 = f"Estimated time: {months} months; cost: ${cost}; average difficulty: {difficulty}."
            line4 = "Recommend doing small projects after each intermediate course to consolidate learning."
            s = '\n'.join([line1, line2, line3, line4])

        brief.append(s)

    return brief
