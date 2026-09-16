import json
import re

from groq import Groq

from app.config import settings
from app.rag.pipeline import answer_question
from eval.answer_tests import ANSWER_TESTS

client = Groq(api_key=settings.groq_api_key)


CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "citation_correct": {"type": "boolean"},
        "citation_completeness": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["citation_correct", "citation_completeness", "reason"],
    "additionalProperties": False,
}


def extract_citations(answer: str):
    matches = re.findall(r"\[Source\s+(\d+)\]", answer)

    return [int(match) for match in matches]


def evaluate_citation(question: str, answer: str, sources: list):
    if not sources:
        return {
            "citation_correct": True,
            "citation_completeness": True,
            "reason": "No sources were returned.",
        }

    source_context = []

    for source in sources:
        source_text = source.get("text", "")

        source_context.append(f"""
[Source {source['source']}]

Filename: {source['filename']}

Pages: {source['page_start']} - {source['page_end']}

Content:

{source_text}
""")

    context = "\n".join(source_context)

    prompt = f"""
You are evaluating citation quality in a document-based RAG system.

Evaluate whether the citations in the answer correctly identify
the source content that supports the answer.

STRICT RULES:

1. Check whether every cited source number actually exists.

2. Check whether the cited source content actually supports
   the factual claim associated with the citation.

3. A citation is correct only if the cited source directly
   supports the claim.

4. Do not assume a source supports a claim just because its
   filename or topic is related.

5. Citation correctness does NOT require exact wording.

6. Citation completeness means factual claims have appropriate
   supporting citations.

7. Claims that are not supported by the provided source content
   should be considered unsupported.

8. Do not use outside knowledge.

9. Do not judge whether the answer is stylistically good.

10. Return ONLY valid JSON matching the provided schema.

QUESTION:

{question}

ANSWER:

{answer}

AVAILABLE SOURCES:

{context}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_effort="low",
        reasoning_format="hidden",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "citation_evaluation",
                "schema": CITATION_SCHEMA,
                "strict": True,
            },
        },
        max_completion_tokens=1024,
    )

    content = response.choices[0].message.content

    return json.loads(content)


def main():
    total = 0
    correct = 0
    complete = 0
    errors = 0

    print("\nCITATION CORRECTNESS EVALUATION")
    print("=" * 80)

    for index, test in enumerate(ANSWER_TESTS, start=1):
        question = test["query"]

        result = answer_question(question=question, top_k=5)

        answer = result["answer"]
        sources = result["sources"]

        citations = extract_citations(answer)

        valid_source_numbers = {source["source"] for source in sources}

        invalid_citations = [
            citation for citation in citations if citation not in valid_source_numbers
        ]

        try:
            if invalid_citations:
                evaluation = {
                    "citation_correct": False,
                    "citation_completeness": False,
                    "reason": (f"Invalid citation numbers: " f"{invalid_citations}"),
                }

            else:
                evaluation = evaluate_citation(
                    question=question, answer=answer, sources=sources
                )

            total += 1

            if evaluation["citation_correct"]:
                correct += 1

            if evaluation["citation_completeness"]:
                complete += 1

            print(f"\n[{index}] {question}")

            print(f"Answer: {answer}")

            print(f"Citations: {citations}")

            print("Correct: " f"{'PASS' if evaluation['citation_correct'] else 'FAIL'}")

            print(
                "Complete: "
                f"{'PASS' if evaluation['citation_completeness'] else 'FAIL'}"
            )

            print(f"Reason: {evaluation['reason']}")

        except Exception as error:
            errors += 1

            print(f"\n[{index}] {question}")
            print(f"JUDGE ERROR: {error}")

    print("\n" + "=" * 80)

    if total > 0:
        correctness_score = correct / total
        completeness_score = complete / total

        print(f"Total queries: {total}")
        print(f"Citation Correct: {correct}/{total}")
        print(f"Citation Completeness: {complete}/{total}")
        print(f"Judge errors: {errors}")

        print("Citation Correctness Score: " f"{correctness_score:.2%}")

        print("Citation Completeness Score: " f"{completeness_score:.2%}")


if __name__ == "__main__":
    main()
