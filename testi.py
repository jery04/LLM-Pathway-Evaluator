from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding = model.encode(
    "Machine Learning",
    normalize_embeddings=True
).tolist()

print(embedding)