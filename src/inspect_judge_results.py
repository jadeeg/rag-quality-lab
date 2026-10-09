import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "judge_results.json"

with RESULTS_PATH.open("r", encoding="utf-8") as file:
    results = json.load(file)

for item in results:
    print("\n" + "=" * 60)
    print("Question:", item["question"])
    print("Category:", item["category"])
    print("Correctness:", item["correctness"], "/ 5")
    print("Groundedness:", item["groundedness"], "/ 5")
    print("Relevance:", item["relevance"], "/ 5")
    print("Abstention:", item["abstention"], "/ 5")
    print("Reason:", item["reason"])

