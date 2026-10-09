from pathlib import Path

document_path = Path("data/documents/vacation_policy.md")

document = document_path.read_text(encoding= "utf-8")

question = "How many vacation days does an employee receive after three years?"

print("Question")
print(question)

print("\nDOCUMENT:")
print(document)

