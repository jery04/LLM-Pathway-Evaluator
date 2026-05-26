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

## Cómo correr (paso a paso)

Sigue estos pasos para crear el entorno virtual `.venv`, instalar dependencias y ejecutar la app. Los ejemplos incluyen PowerShell (Windows), CMD (Windows) y Bash (macOS/Linux).

1) Crear el entorno virtual

- PowerShell (Windows):

```powershell
python -m venv .venv
```

- CMD (Windows):

```cmd
python -m venv .venv
```

- Bash (macOS / Linux):

```bash
python3 -m venv .venv
```

2) Activar el entorno

- PowerShell (Windows):

```powershell
# Si la política de ejecución impide ejecutar scripts, permite temporalmente los scripts para la sesión:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

- CMD (Windows):

```cmd
.\.venv\Scripts\activate.bat
```

- Bash (macOS / Linux):

```bash
source .venv/bin/activate
```

3) Actualizar herramientas de empaquetado e instalar dependencias

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Si ves un error tipo "ModuleNotFoundError: No module named 'pip._internal.cli'", repara pip así (desde el intérprete del `.venv`):

```powershell
& '.venv\Scripts\python.exe' -m ensurepip --upgrade --default-pip
& '.venv\Scripts\python.exe' -m pip install --upgrade pip setuptools wheel
```

4) Ejecutar la aplicación

Tras activar el entorno, ejecuta (ejemplo con Streamlit):

```bash
streamlit run src/app.py
```

O usando el intérprete del `.venv` directamente en Windows:

```powershell
& '.venv\Scripts\python.exe' -m streamlit run src/app.py
```

5) Variables de entorno útiles

- `OPENAI_API_KEY`: clave para la API de OpenAI.
- `HUGGINGFACEHUB_API_TOKEN`: token para Hugging Face Inference API.

Ejemplo (PowerShell):

```powershell
$env:OPENAI_API_KEY = "tu_api_key"
$env:HUGGINGFACEHUB_API_TOKEN = "tu_token"
```

Notas:
- Este repositorio está testeado en Windows; los comandos de Bash sirven para macOS/Linux.
- Si quieres que haga la limpieza del paquete corrupto que genera la advertencia "Ignoring invalid distribution ~ip", dímelo y lo borro del directorio `.venv\Lib\site-packages`.
