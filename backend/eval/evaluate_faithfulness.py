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


def judge_faithfulness(question: str, context: str, answer: str):
    prompt = f"""
You are a strict RAG faithfulness evaluator.

Your task is to classify whether the ANSWER is fully supported
by the provided CONTEXT.

Use ONLY the CONTEXT.

Return:
- FAITHFUL if every factual claim in the answer is supported.
- NOT_FAITHFUL if even one factual claim is not supported.

Do not use outside knowledge.
Ignore writing style.
Ignore usefulness.
Judge only factual grounding.

Return a JSON object with exactly:
{{
  "verdict": "FAITHFUL" or "NOT_FAITHFUL",
  "reason": "short explanation"
}}

QUESTION:
{question}

CONTEXT:
========================
{context}
========================

ANSWER:
{answer}
"""

    try:

        response = judge_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Important for GPT-OSS reasoning models
            reasoning_effort="low",
            reasoning_format="hidden",
            max_completion_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "faithfulness_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": ["FAITHFUL", "NOT_FAITHFUL"],
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


def evaluate_faithfulness():

    total = len(ANSWER_TESTS)

    passed = 0
    failed = 0
    judge_errors = 0

    print("\nRAG FAITHFULNESS EVALUATION")
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
        # 3. Judge faithfulness
        # ---------------------------------------

        judgment = judge_faithfulness(
            question=test["query"], context=context, answer=answer
        )

        # ---------------------------------------
        # 4. Determine result
        # ---------------------------------------

        is_faithful = False
        is_not_faithful = False
        judge_error = False

        judgment_data = None

        # Check for judge error
        if judgment.startswith("JUDGE_ERROR:"):

            judge_error = True

        else:

            try:
                judgment_data = json.loads(judgment)

                verdict = judgment_data.get("verdict", "").upper()

                is_faithful = verdict == "FAITHFUL"

                is_not_faithful = verdict == "NOT_FAITHFUL"

                # Invalid JSON verdict
                if not is_faithful and not is_not_faithful:
                    judge_error = True

            except json.JSONDecodeError:

                judge_error = True

        # ---------------------------------------
        # 5. Update counters + status
        # ---------------------------------------

        if judge_error:

            judge_errors += 1
            status = "JUDGE_ERROR"

        elif is_faithful:

            passed += 1
            status = "PASS"

        elif is_not_faithful:

            failed += 1
            status = "FAIL"

        else:

            judge_errors += 1
            status = "JUDGE_ERROR"

        # ---------------------------------------
        # 6. Print detailed result
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

    print(f"Faithful:         {passed}")

    print(f"Not faithful:     {failed}")

    print(f"Judge errors:     {judge_errors}")

    if valid_judgments > 0:

        faithfulness_score = passed / valid_judgments

        print(f"Faithfulness Score " f"(valid judgments): " f"{faithfulness_score:.2%}")

    else:

        print("Faithfulness Score: " "No valid judge results.")


if __name__ == "__main__":
    evaluate_faithfulness()
