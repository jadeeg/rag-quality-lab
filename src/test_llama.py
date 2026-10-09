import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": (
                "You are evaluating a RAG system. "
                "Reply with exactly: Llama is working."
            ),
        }
    ],
)

print(response.message.content)