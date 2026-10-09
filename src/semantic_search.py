from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

chunks = [
    "N Star Labs provides paid vacation to all full-time employees.",
    "Employees receive 15 vacation days during their first year of employment.",
    "Employees receive 20 vacation days during their second year.",
    "Beginning in the third year of employment, employees receive 25 vacation days per year.",
    "Vacation requests should normally be submitted at least two weeks in advance.",
    "Unused vacation days may be carried over for up to 5 days into the following year.",
    "Managers may deny a vacation request when business-critical staffing requirements make the requested dates impractical."
]

question = "How many vacation days does an employee receive after three years?"

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)
question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

results = sorted(
    zip(chunks, similarities),
    key=lambda x: x[1],
    reverse=True
)

for chunk, score in results:
    print(f"{score:.3f} | {chunk}")