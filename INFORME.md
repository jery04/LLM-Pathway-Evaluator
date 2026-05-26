# Informe técnico

## 1. Objetivo
El proyecto resuelve la exploración de trayectorias profesionales alternativas a partir de un objetivo profesional, habilidades iniciales y preferencias del usuario.

El sistema debe:

- generar varias trayectorias válidas,
- respetar prerrequisitos y restricciones,
- comparar rutas bajo distintos criterios,
- usar un modelo de lenguaje como componente funcional para interpretar y explicar.

## 2. Idea general
El problema se modela como un grafo dirigido de habilidades, cursos y roles.

Cada nodo tiene:

- nombre,
- duración estimada,
- dificultad,
- coste,
- categoría,
- prerequisitos.

Una trayectoria es una secuencia de nodos que parte de las habilidades actuales y llega al rol objetivo.

## 3. Formalización del problema
Sea:

- $S$ el conjunto de habilidades adquiridas,
- $G$ el objetivo profesional,
- $A$ el conjunto de acciones posibles,
- $C$ el conjunto de restricciones.

El estado del sistema se define como:

$$
estado = (S, tiempo, presupuesto)
$$

Una acción consiste en añadir un nodo cuyo conjunto de prerequisitos esté contenido en $S$.

Restricciones usadas:

- prerequisitos obligatorios,
- categoría a evitar,
- coste total,
- complejidad media,
- tiempo total.

## 4. Algoritmo
Se implementó una búsqueda A* con perfiles de preferencia.

La función de coste combina:

- duración del nodo,
- dificultad,
- coste económico,
- sesgo por categoría.

La heurística estima el trabajo restante a partir de los prerequisitos faltantes para alcanzar el objetivo.

Además, se generan perfiles distintos para producir trayectorias alternativas:

- rápida,
- balanceada,
- técnica,
- infraestructura.

## 5. Uso del LLM
El LLM se usa en dos puntos:

1. Interpretación de la intención del usuario en lenguaje natural.
2. Comparación cualitativa de las trayectorias generadas.

Si hay una API de OpenAI o Hugging Face configurada, el sistema la usa. Si no, cae en una lógica local de respaldo.

Variables de entorno admitidas:

- `OPENAI_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN`
- `HUGGINGFACE_MODEL`

## 6. Dataset
El dataset se generó manualmente como una base pequeña pero realista de habilidades y roles tecnológicos.

Incluye categorías como:

- fundamentos,
- datos,
- IA,
- matemáticas,
- cloud,
- DevOps,
- web,
- seguridad.

Esto permite demostrar el sistema sin depender de una fuente externa.

## 7. Interfaz
La interfaz se implementó con Streamlit para facilitar la demo.

Permite:

- escribir el objetivo en lenguaje natural,
- seleccionar habilidades previas,
- generar rutas,
- ver métricas,
- leer la comparación cualitativa.

## 8. Resultados
El prototipo genera trayectorias válidas y distintas para un mismo objetivo.

Cada ruta devuelve:

- secuencia de pasos,
- duración total,
- coste total,
- dificultad media,
- perfil que la originó.

## 9. Cómo ejecutarlo
Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el planificador:

```bash
python src/planner.py
```

Ejecutar la interfaz:

```bash
streamlit run src/app.py
```

## 10. Cronograma sugerido
### Semana 1
- Definir el dominio y el dataset.
- Implementar el grafo y la búsqueda A*.

### Semana 2
- Integrar el LLM.
- Construir la interfaz.
- Probar rutas y casos de uso.

### Semana 3
- Redactar informe final.
- Preparar demo y conclusiones.

## 11. Limitaciones
- El dataset es pequeño y manual.
- La calidad de las trayectorias depende de los pesos elegidos.
- La explicación del LLM depende de la API disponible y de la calidad del modelo.

## 12. Mejoras futuras
- ampliar el dataset,
- añadir más perfiles profesionales,
- exportar resultados a PDF/CSV,
- incorporar evaluación automática de rutas,
- guardar historial de recomendaciones.
