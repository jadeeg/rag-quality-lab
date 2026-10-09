
import json
from pathlib import Path

import ollama


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "rag_outputs.json"
OUTPUT_PATH = ROOT / "results" / "judge_results.json"


def judge_answer(case):
    prompt = f"""
You are an independent evaluator of a RAG system.

Evaluate the answer using only the supplied evidence.

Question:
{case["question"]}

Reference answer:
{case["reference"]}

Retrieved context:
{case["retrieved_context"]}

System answer:
{case["answer"]}

Is the question answerable from the available policy documents?
{case["answerable"]}

Score each dimension from 1 to 5.

CORRECTNESS:
5 = The answer is factually correct and consistent with the reference.
4 = Correct, with a minor omission.
3 = Partially correct.
2 = Mostly incorrect.
1 = Incorrect or misleading.

GROUNDEDNESS:
5 = Every factual claim is supported by the retrieved context.
4 = Supported, with a minor unsupported detail.
3 = Partially supported.
2 = Mostly unsupported.
1 = The answer makes unsupported factual claims.
If the answer simply says it lacks information, do not penalize it
for failing to provide facts that are absent from the context.

RELEVANCE:
5 = Directly addresses the question.
4 = Relevant, with minor unnecessary content.
3 = Partially addresses the question.
2 = Mostly off-topic.
1 = Does not address the question at all.
A concise refusal is relevant when the question cannot be answered.

ABSTENTION:
If answerable is false:
5 = Clearly acknowledges insufficient information without inventing facts.
3 = Expresses uncertainty but is ambiguous.
1 = Invents an answer or falsely claims unsupported information.
If answerable is true:
5 = Provides a correct answer from the evidence.
3 = Partially answers or unnecessarily expresses uncertainty.
1 = Refuses despite sufficient evidence.

FINAL CONSISTENCY CHECK:
Before returning JSON, verify that every score agrees with its explanation.
If the question is unanswerable and the system correctly abstains:
- correctness should be 5 if the refusal is appropriate;
- relevance should be 5 if the refusal addresses the question;
- abstention should be 5;
- groundedness should be 5 if the response makes no unsupported factual claims.
Do not describe a correct refusal as incorrect.

Evaluate the actual answer, not whether the retrieved context is ideal.
Use the reference answer and answerable label as evaluation guidance.
The explanation must be consistent with all scores.
```


Return ONLY valid JSON in this format:
{{
  "correctness": 1,
  "groundedness": 1,
  "relevance": 1,
  "abstention": 1,
  "reason": "Brief explanation"
}}

Do not reward an invented answer.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0}
    )

    return json.loads(response.message.content)


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    results = []

    for index, case in enumerate(cases, start=1):
        print(f"Judging {index}/{len(cases)}: {case['question']}")

        try:
            scores = judge_answer(case)
            results.append({
                "question": case["question"],
                "category": case["category"],
                **scores
            })
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"Could not parse this judgment: {error}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} judgments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()