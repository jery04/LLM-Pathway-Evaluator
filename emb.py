import spacy
from scipy.spatial.distance import cosine
import numpy as np
import langdetect
from langdetect import detect

# Cargar ambos modelos al inicio
nlp_es = spacy.load('es_core_news_md')  # modelo en español
nlp_en = spacy.load('en_core_web_md')   # modelo en inglés

# Función para seleccionar el modelo según el idioma
def seleccionar_modelo(texto):
    try:
        idioma = detect(texto)
        if idioma == 'es':
            return nlp_es
        else:
            return nlp_en
    except:
        print(f"⚠️ No se pudo detectar el idioma de '{texto}', usando modelo EN por defecto")
        return nlp_en

# Textos de ejemplo (puedes cambiar estos valores)
texto1 = "ingeniería informática"
texto2 = "aprender programación"
texto3 = "computer science"
texto4 = "learn programming"

# Seleccionar modelo para cada texto
nlp1 = seleccionar_modelo(texto1)
nlp2 = seleccionar_modelo(texto2)
nlp3 = seleccionar_modelo(texto3)
nlp4 = seleccionar_modelo(texto4)

# Convertir textos a vectores usando sus modelos respectivos
doc1 = nlp1(texto1)
doc2 = nlp2(texto2)
doc3 = nlp3(texto3)
doc4 = nlp4(texto4)

# Calcular similitudes
sim_es_es = doc1.similarity(doc2)      # español vs español
sim_en_en = doc3.similarity(doc4)      # inglés vs inglés
sim_es_en = doc1.similarity(doc3)      # español vs inglés (puede no ser precisa)
