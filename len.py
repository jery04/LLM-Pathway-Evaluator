import json

# Cargar embedding.json
print("=== embedding.json ===")
with open('data/embedding.json', 'r', encoding='utf-8') as f:
    data_embedding = json.load(f)

if isinstance(data_embedding, dict):
    num_elements = len(data_embedding)
    print(f"Tipo: Diccionario")
    print(f"Número de elementos: {num_elements}")
elif isinstance(data_embedding, list):
    num_elements = len(data_embedding)
    print(f"Tipo: Lista")
    print(f"Número de elementos: {num_elements}")

# Cargar normalized_courses.json
print("\n=== normalized_courses.json ===")
with open('data/normalized_courses.json', 'r', encoding='utf-8') as f:
    data_courses = json.load(f)

if isinstance(data_courses, dict):
    num_elements = len(data_courses)
    print(f"Tipo: Diccionario")
    print(f"Número de elementos: {num_elements}")
elif isinstance(data_courses, list):
    num_elements = len(data_courses)
    print(f"Tipo: Lista")
    print(f"Número de elementos: {num_elements}")
