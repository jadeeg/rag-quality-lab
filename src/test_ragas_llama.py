
import asyncio

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


async def main():
    evaluator_llm = llm_factory(
        "llama3.2",
        provider="ollama",
        base_url="http://localhost:11434"
    )

    faithfulness = Faithfulness(llm=evaluator_llm)

    result = await faithfulness.ascore(
        user_input="How many vacation days do employees receive in their third year?",
        response="Employees receive 25 vacation days per year beginning in their third year.",
        retrieved_contexts=[
            "Beginning in the third year of employment, employees receive 25 vacation days per year."
        ]
    )

    print("Faithfulness score:", result.value)
    print("Reason:", result.reason)


if __name__ == "__main__":
    asyncio.run(main())