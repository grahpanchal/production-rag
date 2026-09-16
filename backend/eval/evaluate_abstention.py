from app.rag.pipeline import answer_question

from eval.abstention_tests import ABSTENTION_TESTS

NO_ANSWER = "I couldn't find the answer in the provided documents."


def evaluate_abstention():

    total = len(ABSTENTION_TESTS)

    passed = 0
    failed = 0

    print("\nRAG ABSTENTION EVALUATION")
    print("=" * 80)

    for index, test in enumerate(ABSTENTION_TESTS, start=1):

        # ---------------------------------------
        # 1. Run RAG pipeline
        # ---------------------------------------

        result = answer_question(question=test["query"], top_k=5)

        answer = result["answer"]

        # ---------------------------------------
        # 2. Determine whether system abstained
        # ---------------------------------------

        did_abstain = answer.strip() == NO_ANSWER

        expected_abstain = test["expected_abstain"]

        # ---------------------------------------
        # 3. Compare expected vs actual
        # ---------------------------------------

        if did_abstain == expected_abstain:

            passed += 1
            status = "PASS"

        else:

            failed += 1
            status = "FAIL"

        # ---------------------------------------
        # 4. Print result
        # ---------------------------------------

        print(f"\n[{index}] {status}")

        print(f"Query: {test['query']}")

        print(f"Expected abstain: " f"{expected_abstain}")

        print(f"Actual abstain:   " f"{did_abstain}")

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        if result["sources"]:

            for source in result["sources"]:

                print(
                    f"- {source['filename']} "
                    f"(pages "
                    f"{source['page_start']}-"
                    f"{source['page_end']})"
                )

        else:

            print("- None")

    # -------------------------------------------
    # Final statistics
    # -------------------------------------------

    accuracy = passed / total if total > 0 else 0

    print("\n" + "=" * 80)

    print(f"Total queries:    {total}")

    print(f"Passed:           {passed}")

    print(f"Failed:           {failed}")

    print(f"Abstention Score: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_abstention()
