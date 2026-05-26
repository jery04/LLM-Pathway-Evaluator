# Career Path Explorer AI

Prototipo para el proyecto "Exploración de trayectorias profesionales alternativas".

Requisitos:

- Python 3.10+
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar interfaz Streamlit:

```bash
streamlit run src/app.py
```

Si quieres integrar OpenAI, exporta `OPENAI_API_KEY` en tu entorno.

Para usar un modelo gratuito de Hugging Face, exporta:

```powershell
$env:HUGGINGFACEHUB_API_TOKEN="tu_token"
```

Si quieres probar un modelo ligero localmente, puedes cambiar `HUGGINGFACE_MODEL` por otro modelo compatible con Inference API.
