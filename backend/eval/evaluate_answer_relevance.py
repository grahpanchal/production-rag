import json

from groq import Groq

from app.config import settings
from app.rag.embeddings import create_embeddings
from app.db.qdrant import client
from app.rag.generator import generate_answer

from eval.answer_tests import ANSWER_TESTS

judge_client = Groq(api_key=settings.groq_api_key)


def dense_search(query: str, top_k: int = 5):
    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def judge_answer_relevance(question: str, answer: str):
    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation system.

Your task is to evaluate whether the ANSWER correctly and directly
answers the QUESTION.

Evaluate ONLY answer relevance and correctness.

Rules:

1. Understand what the QUESTION is asking.
2. Check whether the ANSWER directly addresses the question.
3. Check whether the important information required by the question
   is present in the answer.
4. The answer should be factually correct for the question.
5. Do not penalize minor wording differences.
6. Do not penalize formatting or writing style.
7. Do not require the answer to contain every detail from the documents.
8. If the answer is correct and directly answers the question,
   return RELEVANT.
9. If the answer is incorrect, incomplete in an important way,
   or does not answer the question, return NOT_RELEVANT.
10. Do not use outside knowledge to invent facts.

Return a JSON object with exactly:

{{
  "verdict": "RELEVANT" or "NOT_RELEVANT",
  "reason": "short explanation"
}}

QUESTION:
{question}

ANSWER:
{answer}
"""

    try:

        response = judge_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            reasoning_effort="low",
            reasoning_format="hidden",
            max_completion_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_relevance_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": ["RELEVANT", "NOT_RELEVANT"],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["verdict", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = response.choices[0].message.content

        if not content:
            return "JUDGE_ERROR: " "Empty response from judge model."

        return content.strip()

    except Exception as error:

        return "JUDGE_ERROR: " f"{type(error).__name__}: {error}"


def evaluate_answer_relevance():

    total = len(ANSWER_TESTS)

    passed = 0
    failed = 0
    judge_errors = 0

    print("\nRAG ANSWER RELEVANCE EVALUATION")
    print("=" * 80)

    for index, test in enumerate(ANSWER_TESTS, start=1):

        # ---------------------------------------
        # 1. Retrieve context
        # ---------------------------------------

        documents = dense_search(query=test["query"], top_k=5)

        context_parts = []

        for source_index, document in enumerate(documents, start=1):

            payload = document.payload

            context_parts.append(f"""
[Source {source_index}]

Filename: {payload['filename']}
Pages: {payload['page_start']} - {payload['page_end']}

Content:

{payload['text']}
""")

        context = "\n\n".join(context_parts)

        # ---------------------------------------
        # 2. Generate answer
        # ---------------------------------------

        answer = generate_answer(question=test["query"], context=context)

        # ---------------------------------------
        # 3. Judge answer relevance
        # ---------------------------------------

        judgment = judge_answer_relevance(question=test["query"], answer=answer)

        # ---------------------------------------
        # 4. Parse judgment
        # ---------------------------------------

        is_relevant = False
        is_not_relevant = False
        judge_error = False

        judgment_data = None

        if judgment.startswith("JUDGE_ERROR:"):

            judge_error = True

        else:

            try:

                judgment_data = json.loads(judgment)

                verdict = judgment_data.get("verdict", "").upper()

                is_relevant = verdict == "RELEVANT"

                is_not_relevant = verdict == "NOT_RELEVANT"

                if not is_relevant and not is_not_relevant:
                    judge_error = True

            except json.JSONDecodeError:

                judge_error = True

        # ---------------------------------------
        # 5. Update counters
        # ---------------------------------------

        if judge_error:

            judge_errors += 1
            status = "JUDGE_ERROR"

        elif is_relevant:

            passed += 1
            status = "PASS"

        elif is_not_relevant:

            failed += 1
            status = "FAIL"

        else:

            judge_errors += 1
            status = "JUDGE_ERROR"

        # ---------------------------------------
        # 6. Print result
        # ---------------------------------------

        print(f"\n[{index}] {status}")

        print(f"Query: {test['query']}")

        print("\nAnswer:")
        print(answer)

        print("\nJudge:")

        if judgment_data:

            print(f"Verdict: " f"{judgment_data.get('verdict')}")

            print(f"Reason: " f"{judgment_data.get('reason')}")

        else:

            print(judgment)

    # -------------------------------------------
    # Final statistics
    # -------------------------------------------

    valid_judgments = total - judge_errors

    print("\n" + "=" * 80)

    print(f"Total queries:    {total}")

    print(f"Relevant:         {passed}")

    print(f"Not relevant:     {failed}")

    print(f"Judge errors:     {judge_errors}")

    if valid_judgments > 0:

        relevance_score = passed / valid_judgments

        print(
            f"Answer Relevance Score " f"(valid judgments): " f"{relevance_score:.2%}"
        )

    else:

        print("Answer Relevance Score: " "No valid judge results.")


if __name__ == "__main__":
    evaluate_answer_relevance()
