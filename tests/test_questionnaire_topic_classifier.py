from unittest.mock import MagicMock

from core_lib.classification import QuestionnaireTopicClassifier, TopicProfile


def test_topic_profile_fingerprint_is_stable_for_equivalent_whitespace():
    first = TopicProfile(
        description="Purchase   requisition forms",
        primary_topics=["Purchase requisitions"],
        confidence=0.9,
    )
    second = TopicProfile(
        description="Purchase requisition forms",
        primary_topics=["Purchase requisitions"],
        confidence=0.2,
    )
    assert first.fingerprint == second.fingerprint


def test_topic_profile_fingerprint_is_order_and_case_independent():
    first = TopicProfile(
        description="Purchase requisition forms",
        primary_topics=["Approvals", "Purchase Requisitions"],
        product_areas=["eProcurement", "CloudSync Enterprise"],
        capabilities=["Custom Forms", "Workflow Rules"],
        document_category=" Product Documentation ",
        language=" EN ",
    )
    second = TopicProfile(
        description=" purchase   requisition forms ",
        primary_topics=["purchase requisitions", "approvals"],
        product_areas=["cloudsync enterprise", "eprocurement"],
        capabilities=["workflow rules", "custom forms"],
        document_category="product documentation",
        language="en",
    )

    assert first.fingerprint == second.fingerprint


def test_topic_profile_fingerprint_changes_with_scope():
    requisitions = TopicProfile(description="Purchase requisitions")
    invoicing = TopicProfile(description="E-invoicing")
    assert requisitions.fingerprint != invoicing.fingerprint


def test_even_question_sampling_covers_complete_sheet():
    questions = [f"Question {index}" for index in range(100)]
    sampled = QuestionnaireTopicClassifier.sample_questions(questions, limit=5)
    assert sampled[0] == "Question 0"
    assert sampled[-1] == "Question 99"
    assert any(item in sampled for item in ("Question 49", "Question 50"))


def test_question_sampling_handles_small_limits_without_dividing_by_zero():
    questions = ["First", "Second", "Third"]

    assert QuestionnaireTopicClassifier.sample_questions(questions, limit=1) == ["First"]
    assert QuestionnaireTopicClassifier.sample_questions(questions, limit=0) == []


def test_classifier_returns_structured_profile():
    expected = TopicProfile(
        description="Questions about purchase requisition creation and approval.",
        primary_topics=["purchase requisitions"],
        product_areas=["eProcurement"],
        capabilities=["form customization"],
        document_category="sales_rfx",
        language="en",
        confidence=0.93,
    )
    client = MagicMock()
    client.chat.return_value = {"content": expected.model_dump()}
    classifier = QuestionnaireTopicClassifier(max_questions=5)
    classifier._client = client

    result = classifier.classify(
        workbook_name="rfx.xlsx",
        sheet_name="eProcurement",
        questions=["Can requisition forms be customized?"],
        language="en",
    )

    assert result.primary_topics == ["purchase requisitions"]
    assert result.product_areas == ["eProcurement"]
    assert result.detection_method == "llm"
    assert result.classifier_version == "questionnaire-topic-v1"
    assert result.fingerprint


def test_classifier_failure_returns_neutral_default():
    client = MagicMock()
    client.chat.side_effect = RuntimeError("offline")
    classifier = QuestionnaireTopicClassifier()
    classifier._client = client
    result = classifier.classify("rfq.xlsx", "Purchasing", ["Question?"])
    assert result.confidence == 0.0
    assert result.detection_method == "default"


def test_classifier_preserves_explicit_product_name_when_llm_generalizes_it():
    client = MagicMock()
    client.chat.return_value = {
        "content": TopicProfile(
            description="A questionnaire about multi-cloud management.",
            product_areas=["Multi-cloud orchestration platforms"],
            confidence=0.9,
        ).model_dump()
    }
    classifier = QuestionnaireTopicClassifier()
    classifier._client = client

    result = classifier.classify(
        "rfx.xlsx",
        "Cloud",
        [
            "For CloudSync Enterprise, describe migration capabilities.",
            "How many resources can CloudSync Enterprise manage?",
        ],
    )

    assert "CloudSync Enterprise" in result.product_areas


def test_classifier_re_normalizes_product_areas_after_merging_explicit_names():
    client = MagicMock()
    client.chat.return_value = {
        "content": TopicProfile(
            description="CloudSync Enterprise questionnaire.",
            product_areas=[f"Area {index}" for index in range(20)],
            confidence=0.9,
        ).model_dump()
    }
    classifier = QuestionnaireTopicClassifier()
    classifier._client = client

    result = classifier.classify(
        "rfx.xlsx",
        "Cloud",
        ["For CloudSync Enterprise, describe migration capabilities."],
    )

    assert result.product_areas[0] == "CloudSync Enterprise"
    assert len(result.product_areas) == 20
    assert len({value.casefold() for value in result.product_areas}) == len(result.product_areas)
