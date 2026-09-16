from app.rag.pipeline import answer_question
from eval.answer_tests import ANSWER_TESTS


def evaluate_answers():
    total = len(ANSWER_TESTS)
    passed = 0

    print("\nRAG ANSWER QUALITY EVALUATION")
    print("=" * 70)

    for index, test in enumerate(ANSWER_TESTS, start=1):

        result = answer_question(question=test["query"], top_k=5)

        answer = result["answer"]
        answer_lower = answer.lower()

        expected_keywords = test["expected_keywords"]

        keyword_match = all(
            keyword.lower() in answer_lower for keyword in expected_keywords
        )

        has_citation = "[source" in answer_lower

        source_match = any(
            source["filename"] == test["expected_filename"]
            for source in result["sources"]
        )

        passed_test = keyword_match and has_citation and source_match

        if passed_test:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{index}] {status}")
        print(f"Query: {test['query']}")
        print(f"Answer: {answer}")
        print(f"Expected keywords: {expected_keywords}")
        print(f"Citation present: {has_citation}")
        print(f"Correct source: {source_match}")

    score = passed / total

    print("\n" + "=" * 70)
    print(f"Answer Quality Score: {score:.2%}")
    print(f"Passed: {passed}/{total}")


if __name__ == "__main__":
    evaluate_answers()
