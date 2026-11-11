import os
import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GeminiModel

from app.chatbot import ask_customer_support
from tests.test_cases import test_cases

load_dotenv()

gemini_model = GeminiModel(
    model=os.getenv("GEMINI_MODEL"), api_key=os.getenv("GOOGLE_API_KEY")
)

faithfulness = FaithfulnessMetric(threshold=0.7, model=gemini_model)

answer_relevency = AnswerRelevancyMetric(threshold=0.7, model=gemini_model)

policy_compliance = GEval(
    name="Policy Compliance",
    criteria="""
         Evaluate whether the assistant follows the company policies.

    The assistant must not:
    1. Invent return or refund policies.
    2. Promise exceptions not mentioned in the context.
    3. Provide information that contradicts the policy.

    Give a high score when the answer follows these rules.
    """,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=gemini_model,
)


@pytest.mark.parametrize("test_data", test_cases)
def test_return_policy(test_data):
    result = ask_customer_support(test_data["input"])

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        expected_output=test_data["expected_behavior"],
        retrieval_context=result["retrieval_context"],
    )

    assert_test(test_case, [faithfulness, answer_relevency, policy_compliance])
