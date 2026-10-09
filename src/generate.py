import ollama


def generate_answer(question, context):
    prompt = f"""
You are a support assistant for Northstar Labs.

Answer the question using ONLY the context below.

Rules:
- Do not invent policies, numbers, or facts.
- If the context does not contain the answer, say you do not have enough information.
- Give a concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response.message.content


if __name__ == "__main__":
    question = (
        "How many vacation days does an employee receive "
        "during their third year?"
    )

    context = (
        "Beginning in the third year of employment, "
        "employees receive 25 vacation days per year."
    )

    answer = generate_answer(question, context)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)