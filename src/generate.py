def build_prompt(question, context):
    return f"""
You are an assistant for N Labs.

Answer the question using ONLY the information provided
in the context.

If the context does not contain enough information,
say that you do not have enough information.

Do not invent policies, numbers, dates, or rules.

Context:
{context}

Question:
{question}

Answer:
"""