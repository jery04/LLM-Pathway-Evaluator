# test_planner_methods.py
"""Casos de prueba para _is_skill_covered y _passes_category_filter"""

# test_planner_methods.py
import sys
from pathlib import Path

# Agregar la carpeta src al path de Python
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))


from planner import (
    _is_skill_covered,
    _passes_category_filter,
    _normalize_token,
    _cosine_similarity,
    SIMILARITY_THRESHOLD,
    CATEGORY_SIMILARITY_THRESHOLD,
    load_courses,
    index_courses,
    _load_embeddings_data
)
from typing import List, Dict, Optional


def test_is_skill_covered():
    """Casos de prueba para _is_skill_covered"""
    
    print("\n" + "="*60)
    print("TEST: _is_skill_covered")
    print("="*60)
    
    # Caso 1: Skill vacío o None
    print("\n📋 Caso 1: Skill vacío")
    result = _is_skill_covered("", ["Python"])
    print(f"   skill='', known=['Python'] -> {result} (esperado: True)")
    assert result == True, "Fallo: skill vacío debe retornar True"
    
    # Caso 2: Known skills vacío
    print("\n📋 Caso 2: Known skills vacío")
    result = _is_skill_covered("Machine Learning", [])
    print(f"   skill='Machine Learning', known=[] -> {result}")
    # Nota: Depende de get_text_embedding, puede ser True o False
    
    # Caso 3: Match exacto (lexical)
    print("\n📋 Caso 3: Match exacto lexical")
    result = _is_skill_covered("python", ["python", "java"])
    print(f"   skill='python', known=['python','java'] -> {result} (esperado: True)")
    assert result == True, "Fallo: match exacto debe retornar True"
    
    # Caso 4: Skill contenido en known (substring)
    print("\n📋 Caso 4: Substring match")
    result = _is_skill_covered("py", ["python programming"])
    print(f"   skill='py', known=['python programming'] -> {result} (esperado: True)")
    assert result == True, "Fallo: substring debe retornar True"
    
    # Caso 5: Known contenido en skill
    print("\n📋 Caso 5: Known como substring de skill")
    result = _is_skill_covered("advanced python", ["python"])
    print(f"   skill='advanced python', known=['python'] -> {result} (esperado: True)")
    assert result == True, "Fallo: known como substring debe retornar True"
    
    # Caso 6: Sin match lexical pero similar semánticamente
    print("\n📋 Caso 6: Similaridad semántica")
    # "programming" y "coding" deberían tener alta similitud coseno
    result = _is_skill_covered("programming", ["coding"])
    print(f"   skill='programming', known=['coding'] -> {result}")
    print(f"   (depende del threshold {SIMILARITY_THRESHOLD} y embeddings)")
    
    # Caso 7: Skills completamente diferentes
    print("\n📋 Caso 7: Skills diferentes")
    result = _is_skill_covered("physics", ["cooking"])
    print(f"   skill='physics', known=['cooking'] -> {result} (esperado: False)")
    
    # Caso 8: Múltiples known skills
    print("\n📋 Caso 8: Múltiples known skills")
    result = _is_skill_covered("data analysis", ["python", "statistics", "sql"])
    print(f"   skill='data analysis', known=['python','statistics','sql'] -> {result}")
    
    # Caso 9: Normalización de tokens
    print("\n📋 Caso 9: Normalización de tokens")
    result = _is_skill_covered("  PyThOn  ", ["  python  "])
    print(f"   skill='  PyThOn  ', known=['  python  '] -> {result} (esperado: True)")
    assert result == True, "Fallo: normalización debe retornar True"
    
    print("\n✅ Tests de _is_skill_covered completados")


def test_passes_category_filter():
    """Casos de prueba para _passes_category_filter"""
    
    print("\n" + "="*60)
    print("TEST: _passes_category_filter")
    print("="*60)
    
    # Cargar datos reales para pruebas
    courses = load_courses()
    course_index = index_courses(courses)
    embeddings_data = _load_embeddings_data() or {}
    
    # Buscar algunos cursos reales
    sample_courses = []
    for name in list(course_index.keys())[:5]:  # Tomar primeros 5 cursos
        sample_courses.append(course_index[name])
    
    if not sample_courses:
        print("⚠️ No se pudieron cargar cursos para pruebas")
        return
    
    # Caso 1: Sin categorías a evitar
    print("\n📋 Caso 1: avoid_categories = None")
    course = sample_courses[0]
    result = _passes_category_filter(course, None, embeddings_data)
    print(f"   curso='{course.get('name')}', avoid=None -> {result} (esperado: True)")
    assert result == True, "Fallo: avoid=None debe retornar True"
    
    # Caso 2: avoid_categories vacío
    print("\n📋 Caso 2: avoid_categories = []")
    result = _passes_category_filter(course, [], embeddings_data)
    print(f"   curso='{course.get('name')}', avoid=[] -> {result} (esperado: True)")
    assert result == True, "Fallo: avoid=[] debe retornar True"
    
    # Caso 3: Categoría a evitar que NO coincide con el curso
    print("\n📋 Caso 3: Categoría que NO coincide")
    result = _passes_category_filter(course, ["cooking", "art"], embeddings_data)
    print(f"   curso='{course.get('name')}', avoid=['cooking','art'] -> {result} (esperado: True)")
    
    # Caso 4: Curso sin nombre
    print("\n📋 Caso 4: Curso sin nombre")
    invalid_course = {"name": "", "category": "Data"}
    result = _passes_category_filter(invalid_course, ["Data"], embeddings_data)
    print(f"   curso sin nombre, avoid=['Data'] -> {result} (esperado: True)")
    assert result == True, "Fallo: curso sin nombre debe retornar True"
    
    # Caso 5: Curso sin embedding
    print("\n📋 Caso 5: Curso sin embedding disponible")
    fake_course = {"name": "Curso Inventado XYZ 123", "category": "Fake"}
    result = _passes_category_filter(fake_course, ["Fake"], embeddings_data)
    print(f"   curso sin embedding, avoid=['Fake'] -> {result} (esperado: True)")
    assert result == True, "Fallo: curso sin embedding debe retornar True"
    
    # Caso 6: Categorías normalizadas
    print("\n📋 Caso 6: Normalización de categorías")
    result = _passes_category_filter(course, ["  DATA  ", "  AI  "], embeddings_data)
    print(f"   curso='{course.get('name')}', avoid=['  DATA  ','  AI  '] -> {result}")
    
    # Caso 7: Simulación de filtro positivo (categoría similar)
    print("\n📋 Caso 7: Categoría similar al curso (debería filtrar si supera threshold)")
    # Este caso depende de los embeddings reales
    # Buscar un curso y probar con su propia categoría
    if embeddings_data:
        for course in sample_courses:
            course_name = course.get('name')
            if course_name and course_name in embeddings_data:
                course_category = course.get('category', '').strip()
                if course_category:
                    result = _passes_category_filter(course, [course_category], embeddings_data)
                    print(f"   curso='{course_name}', avoid=['{course_category}'] -> {result}")
                    print(f"   (si similitud >= {CATEGORY_SIMILARITY_THRESHOLD} -> False, sino True)")
                    break
    
    print("\n✅ Tests de _passes_category_filter completados")


def test_edge_cases():
    """Casos borde adicionales"""
    
    print("\n" + "="*60)
    print("TEST: Casos borde")
    print("="*60)
    
    # Caso borde 1: Known skills con valores None
    print("\n📋 Borde 1: Known skills con None")
    try:
        result = _is_skill_covered("Python", [None, "java", ""])
        print(f"   skill='Python', known=[None, 'java', ''] -> {result}")
        print("   ✅ No lanzó excepción")
    except Exception as e:
        print(f"   ❌ Falló con excepción: {e}")
    
    # Caso borde 2: Todos los known skills son None o vacíos
    print("\n📋 Borde 2: Todos known skills inválidos")
    try:
        result = _is_skill_covered("Python", [None, "", "   "])
        print(f"   skill='Python', known=[None, '', '   '] -> {result}")
    except Exception as e:
        print(f"   ❌ Falló con excepción: {e}")
    
    # Caso borde 3: avoid_categories con valores None
    print("\n📋 Borde 3: avoid_categories con None")
    course = {"name": "Test Course", "category": "Test"}
    embeddings_data = {}
    result = _passes_category_filter(course, [None, "test", ""], embeddings_data)
    print(f"   avoid=[None, 'test', ''] -> {result} (esperado: True o filtro si aplica)")
    
    # Caso borde 4: Caracteres especiales en skills
    print("\n📋 Borde 4: Caracteres especiales")
    result = _is_skill_covered("C++", ["c++", "programming"])
    print(f"   skill='C++', known=['c++','programming'] -> {result} (esperado: True)")
    
    result = _is_skill_covered("C#", ["csharp", "dotnet"])
    print(f"   skill='C#', known=['csharp','dotnet'] -> {result}")
    
    print("\n✅ Tests de casos borde completados")


def test_integration_with_real_data():
    """Prueba de integración con datos reales"""
    
    print("\n" + "="*60)
    print("TEST: Integración con datos reales")
    print("="*60)
    
    # Cargar datos reales
    courses = load_courses()
    print(f"📚 Cursos cargados: {len(courses)}")
    
    if not courses:
        print("⚠️ No se pudieron cargar cursos para prueba de integración")
        return
    
    # Usar un curso real para probar el filtro de categorías
    embeddings_data = _load_embeddings_data()
    print(f"📊 Embeddings cargados: {len(embeddings_data) if embeddings_data else 0}")
    
    # Probar con algunos cursos reales
    test_courses = courses[:3]
    
    for i, course in enumerate(test_courses):
        print(f"\n🏫 Curso {i+1}: {course.get('name', 'Sin nombre')}")
        print(f"   Categoría: {course.get('category', 'Sin categoría')}")
        
        # Probar filtro con categoría propia
        avoid = [course.get('category', '')]
        result = _passes_category_filter(course, avoid, embeddings_data or {})
        print(f"   Filtro con propia categoría: {result}")
        
        # Probar filtro con categoría diferente
        result2 = _passes_category_filter(course, ["cocina", "arte", "música"], embeddings_data or {})
        print(f"   Filtro con categorías no relacionadas: {result2}")
    
    print("\n✅ Tests de integración completados")


if __name__ == "__main__":
    print("\n" + "🔬 INICIANDO PRUEBAS UNITARIAS 🔬")
    
    # Ejecutar todas las pruebas
    test_is_skill_covered()
    test_passes_category_filter()
    test_edge_cases()
    test_integration_with_real_data()
    
    print("\n" + "="*60)
    print("🎉 TODAS LAS PRUEBAS COMPLETADAS 🎉")
    print("="*60)