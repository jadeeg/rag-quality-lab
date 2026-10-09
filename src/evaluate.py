from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI


data = {
    "question": [
        "How many vacation days does an employee receive during their first year?",
        "How many vacation days does an employee receive during their third year?",
        "How far in advance should vacation normally be requested?",
        "How many unused vacation days can be carried over?",
        "What happens if business-critical staffing requirements make the requested vacation dates impractical?"
    ],

    "contexts": [
        [
            "Employees receive 15 vacation days during their first year of employment."
        ],
        [
            "Beginning in the third year of employment, employees receive 25 vacation days per year."
        ],
        [
            "Vacation requests should normally be submitted at least two weeks in advance."
        ],
        [
            "Unused vacation days may be carried over for up to 5 days into the following year."
        ],
        [
            "Managers may deny a vacation request when business-critical staffing requirements make the requested dates impractical."
        ]
    ],

    "answer": [
        "Employees receive 15 vacation days during their first year.",
        "Employees receive 25 vacation days during their third year.",
        "Vacation should normally be requested at least two weeks in advance.",
        "Employees can carry over up to 5 unused vacation days.",
        "A manager may deny the vacation request."
    ],

    "reference": [
        "Employees receive 15 vacation days during their first year.",
        "Employees receive 25 vacation days per year beginning in the third year.",
        "Vacation requests should normally be submitted at least two weeks in advance.",
        "Up to 5 unused vacation days can be carried over into the following year.",
        "A manager may deny the vacation request."
    ]
}

dataset = Dataset.from_dict(data)


evaluator_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

evaluator_llm = LangchainLLMWrapper(evaluator_llm)


result = evaluate(
    dataset,
    metrics=[
        Faithfulness()
    ],
    llm=evaluator_llm
)

print(result)