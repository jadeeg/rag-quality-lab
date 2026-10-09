
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from generate import generate_answer


chunks = [
    "N Star Labs provides paid vacation to all full-time employees.",
    "Employees receive 15 vacation days during their first year of employment.",
    "Employees receive 20 vacation days during their second year.",
    "Beginning in the third year of employment, employees receive 25 vacation days per year.",
    "Vacation requests should normally be submitted at least two weeks in advance.",
    "Unused vacation days may be carried over for up to 5 days into the following year.",
    "Managers may deny a vacation request when business-critical staffing requirements make the requested dates impractical."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)



def retrieve(question, top_k=3, threshold=0.35):
    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    ranked = sorted(
        zip(chunks, similarities),
        key=lambda item: item[1],
        reverse=True
    )

   
    filtered = [
        (chunk, score)
        for chunk, score in ranked
        if score >= threshold
    ]

    return filtered[:top_k]


def answer_question(question):
    results = retrieve(question)

    context = "\n".join(
        chunk for chunk, score in results
    )

    answer = generate_answer(question, context)

    return {
        "question": question,
        "retrieved_context": context,
        "answer": answer
    }


if __name__ == "__main__":
    question = (
        "How many vacation days does an employee receive "
        "during their third year?"
    )

    result = answer_question(question)

    print("QUESTION:")
    print(result["question"])

    print("\nRETRIEVED CONTEXT:")
    print(result["retrieved_context"])
    

    print("\nLLAMA ANSWER:")
    print(result["answer"])
    
    
    