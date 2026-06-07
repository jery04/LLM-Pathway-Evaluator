# LLM Pathway Evaluator 🚀

**Custom Learning Path Generation for Online Course Catalogs**

This system solves a constrained planning and optimization problem in the professional learning domain. Given a set of candidate decisions—courses, skills, certifications—it generates multiple valid learning pathways and evaluates them across technical metrics such as cost, duration, difficulty, and semantic alignment. The architecture uses a Large Language Model (LLM) as an analysis engine to compare, rank, and explain alternative progression strategies.

## Overview 🔍

The project converts normalized online course catalogs into structured learning trajectories. Using normalized course metadata and NLP embeddings, the system:

- 📥 Extracts and normalizes online course information.
- 🧠 Computes semantic embeddings and similarity scores.
- 🛤️ Builds learning pathways that connect skills, objectives, and prerequisites.
- 🌐 Presents results through a lightweight Streamlit web UI.

The goal is to help learners, professionals, and teams identify the most appropriate sequence of courses based on goals and prior experience.

## Project Structure 🗂️

```
LLM Pathway Evaluator/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── data/
│   ├── embedding.json         # Cached course embeddings
│   ├── normalized_courses.json # Normalized course metadata
│   └── csv/                   # Raw dataset CSV files
│       ├── dataset_1.csv
│       ├── dataset_2.csv
│       └── ...
├── src/
│   ├── app.py                 # Streamlit UI for pathway exploration
│   ├── download_dataset.py    # Kaggle dataset download and normalization
│   ├── llm_adapter.py         # LLM / spaCy adapter for embeddings and explanations
│   └── planner.py             # Path planning and course modeling logic
│
```

## Installation 🛠️

### Prerequisites 📌

- Python 3.8+
- pip
- Internet access for dependency installation and spaCy model downloads

### 1. Clone the repository 📂

```bash
git clone https://github.com/jery04/LLM-Pathway-Evaluator.git
cd "LLM Pathway Evaluator"
```

### 2. Create and activate a virtual environment 🧰

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use cmd:

```cmd
.\.venv\Scripts\activate.bat
```

If you use Git Bash or WSL:

```bash
source .venv/bin/activate
```

### 3. Install dependencies 📦

```bash
pip install -r requirements.txt
```

### 4. Prepare dataset files 📁

Verify the presence of the following data artifacts:

- `data/normalized_courses.json`
- `data/embedding.json`
- `data/csv/*.csv`

If the files are missing, regenerate them with:

```bash
python src/download_dataset.py
```

### 5. Launch the web UI (inside the created environment) 🚀

```bash
streamlit run src/app.py
```

## Key Dependencies 📘

- `streamlit`: interactive web UI for pathway visualization
- `google-genai`: Google Gemini client for LLM text generation
- `spacy`: NLP pipeline and embeddings
- `langdetect`: automatic language detection for spaCy model selection
- `kagglehub` / `kagglesdk`: Kaggle dataset download support
- `numpy` / `scipy`: numerical computation and vector similarity

## Example Output 🎯

A typical generated pathway might look like this:

- Target: Learn machine learning with Python
- Proposed pathway:
  1. Course: Python Fundamentals
  2. Course: Statistics for Data Science
  3. Course: Machine Learning with scikit-learn
  4. Course: Data Analysis Project

> The engine builds a coherent progression based on semantic relationships among courses, skills, and learning objectives.

---

Built with ❤️ by a human 


