
import json
from pathlib import Path

from rag import answer_question


ROOT = Path(__file__).resolve().parent.parent

dataset_path = ROOT / "data" /"documents"/ "evaluation_dataset.json"
output_path = ROOT / "results" / "rag_outputs.json"

print("Project root:", ROOT)
print("Dataset path:", dataset_path)
print("Dataset exists:", dataset_path.exists())

with dataset_path.open("r", encoding="utf-8") as file:
    evaluation_cases = json.load(file)

results = []

for case in evaluation_cases:
    print(f"Evaluating: {case['question']}")

    rag_result = answer_question(case["question"])

    results.append({
        **case,
        "retrieved_context": rag_result["retrieved_context"],
        "answer": rag_result["answer"]
    })

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print(f"\nCompleted {len(results)} test cases.")
print(f"Results saved to: {output_path}")