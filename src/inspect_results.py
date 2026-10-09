
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
results_path = ROOT / "results" / "rag_outputs.json"

with results_path.open("r", encoding="utf-8") as file:
    results = json.load(file)

for i, item in enumerate(results, start=1):
    print(f"\n{'=' * 60}")
    print(f"TEST {i}: {item['category']}")
    print(f"QUESTION: {item['question']}")
    print(f"EXPECTED: {item['reference']}")
    print(f"LLAMA: {item['answer']}")
    print(f"ANSWERABLE: {item['answerable']}")
    print(f"RETRIEVED CONTEXT:\n{item['retrieved_context']}")