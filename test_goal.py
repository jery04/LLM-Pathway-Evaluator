import src.planner as planner
import src.llm_adapter as llm

user_text = 'I want to learn web design.'
parsed = llm.parse_input(user_text)
print('PARSED:', parsed)
goal = parsed.get('goal') or user_text
print('GOAL USED:', goal)

courses = planner.load_courses()
paths = planner.generate_paths(courses, ['HTML'], goal, max_paths=3)
print('NUM PATHS=', len(paths))
for i,p in enumerate(paths,1):
    print('PATH', i, '->', ' -> '.join(p.get('path', [])))

print('\nEXPLANATION:\n')
print(llm.explain_comparison(paths, {'goal': goal, 'skills': ['HTML'], 'weekly_time': 8}))
