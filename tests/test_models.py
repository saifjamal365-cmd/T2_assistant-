"""Every offered model, run through the same real answer-writing test battery.

Not just listed on Groq's docs - actually exercised through this app's own
answer prompt (`answer._write_answer`), since a model can look fine on a
throwaway "say hi" check and still fail here. This file is what caught the
concrete failures behind AVAILABLE_MODELS's current shortlist (see the
comment in agents/llm.py): a model that couldn't do the tool-calling the
router needs, one that degraded into repeating its own reference material on
a harder question, one that silently ignored an Arabic question and answered
in English, and one that fabricated an unrelated policy in the wrong
language entirely. Run this (`pytest -m integration tests/test_models.py`)
before adding any model to AVAILABLE_MODELS.
"""

from __future__ import annotations

import pytest

from t2_assistant.agents import answer
from t2_assistant.agents.llm import AVAILABLE_MODELS
from t2_assistant.knowledge.search import Passage

_LEAVE_PASSAGE = [
    Passage(
        text="Full-time employees receive 25 paid annual leave days per year.",
        doc_id="HR-0008",
        title="Annual Leave Policy",
        department="Human Resources",
        doc_type="Policy",
        version="2.0",
        score=0.9,
    )
]
_LEAVE_PASSAGE_AR = [
    Passage(
        text="يحصل الموظفون بدوام كامل على 25 يوم إجازة سنوية مدفوعة الأجر سنويًا.",
        doc_id="HR-0008",
        title="سياسة الإجازة السنوية",
        department="Human Resources",
        doc_type="Policy",
        version="2.0",
        score=0.9,
    )
]
_SALARY_PASSAGES = [
    Passage(
        text="Salaries are paid monthly on the 25th. No early withdrawal is permitted "
        "except via the exception process in HR-0127.",
        doc_id="FIN-0022",
        title="Payment Policy",
        department="Finance",
        doc_type="Procedure",
        version="3.0",
        score=0.9,
    ),
    Passage(
        text="Employees facing hardship may request an early salary withdrawal "
        "exception via HR Operations, subject to manager approval.",
        doc_id="HR-0127",
        title="Early Salary Withdrawal Exception",
        department="Human Resources",
        doc_type="Procedure",
        version="3.0",
        score=0.88,
    ),
]

# A sane answer is short and direct, not a repeat of its own reference
# material - the real failure this guards against ran for several paragraphs
# once it started copying the passages fed to it back into the reply.
_MAX_SANE_ANSWER_CHARS = 600


@pytest.mark.integration
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_model_answers_an_english_question_correctly(model: str) -> None:
    reply = answer._write_answer("How many annual leave days do I get?", _LEAVE_PASSAGE, "", model)
    assert "25" in reply
    assert "HR-0008" in reply
    assert len(reply) < _MAX_SANE_ANSWER_CHARS


@pytest.mark.integration
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_model_answers_an_arabic_question_correctly(model: str) -> None:
    reply = answer._write_answer(
        "كم عدد أيام الإجازة السنوية التي أحصل عليها؟", _LEAVE_PASSAGE_AR, "", model
    )
    assert "25" in reply
    assert "HR-0008" in reply
    assert len(reply) < _MAX_SANE_ANSWER_CHARS


@pytest.mark.integration
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_model_handles_a_harder_multi_passage_question(model: str) -> None:
    reply = answer._write_answer(
        "how can get my salary before his period come", _SALARY_PASSAGES, "", model
    )
    assert not answer._is_decline(reply)
    assert "HR-0127" in reply
    assert len(reply) < _MAX_SANE_ANSWER_CHARS


@pytest.mark.integration
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
def test_model_declines_when_the_passages_dont_cover_it(model: str) -> None:
    reply = answer._write_answer(
        "Can I buy a pet dragon with the company credit card?", _LEAVE_PASSAGE, "", model
    )
    assert answer._is_decline(reply)
