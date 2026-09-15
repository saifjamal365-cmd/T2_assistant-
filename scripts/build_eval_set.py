"""Build the Phase-8 evaluation dataset (500 questions).

The whole point of an evaluation set is that it must be *fair*: the questions
and their correct answers cannot be invented by the same process that grades
them. So every answerable question here is derived from the knowledge-base
generator's own source data (scripts/generate_kb.py) - the same topics,
the same seeded parameter values, the same FAQ text used to build the
documents themselves. Nothing here is hand-typed as "the right answer";
it is read out of the one place the right answer actually lives.

The set has three parts:

  A. FAQ questions, as written        - 96 topics-with-FAQs x 2 languages = 192
  B. The same questions, reworded     - a different phrasing, same fact/doc  = 192
  C. Deliberately unanswerable        - hand-written, not in the corpus     = 116
                                                                      total = 500

(A) and (B) test retrieval and answer correctness. (C) tests honesty - that
the assistant says "I don't know" instead of guessing (NFR-05). (C) mixes
"hard" negatives (a real-sounding HR/IT/Finance question the corpus simply
does not cover) with "easy" ones (obviously out of scope), so the honesty
score is not just measuring how easy the negatives were.

A topic (e.g. "annual leave") has up to eight documents at Head Office - four
types (Policy, Procedure, FAQ, Quick Reference) in two languages - that all
describe the same rules, and retrieval is cross-lingual by design (Section 9).
The answer agent may ground a reply in whichever document its retrieval step
returns, and citing the Policy document - or the other language's version -
instead of the FAQ the question text came from is just as correct. So each
item records `expected_docs`: every Head-Office document for that topic, and
grading checks whether the reply cites any document from that set.

Run:  python scripts/build_eval_set.py
Writes: eval/dataset.json
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_kb as gen  # noqa: E402  (needs the path insert above)

KB_DIR = ROOT / "data" / "knowledge_base"
OUT_PATH = ROOT / "eval" / "dataset.json"

_OFFICE_NAME = {"en": "Head Office", "ar": gen.OFFICES[0][2]}

# ---- (A)/(B): natural-language wrappers used to reword a question ---------
# Applied deterministically (by index), not by guesswork - so results are
# reproducible and every question still means exactly the same thing.
_PARAPHRASE_EN = [
    lambda q: f"Could you tell me {q[0].lower()}{q[1:-1]}?",
    lambda q: f"I have a question: {q}",
    lambda q: f"{q[:-1]}, can you explain?",
    lambda q: f"Quick question - {q[0].lower()}{q[1:-1]}?",
]
_PARAPHRASE_AR = [
    lambda q: f"هل يمكنك أن تخبرني: {q}",
    lambda q: f"لدي سؤال: {q}",
    lambda q: f"{q} أرجو التوضيح.",
    lambda q: f"سؤال سريع - {q}",
]


def _paraphrase(question: str, lang: str, index: int) -> str:
    templates = _PARAPHRASE_EN if lang == "en" else _PARAPHRASE_AR
    return templates[index % len(templates)](question)


# ---- (C): deliberately unanswerable questions ------------------------------
# Paired 1:1, English then the matching Arabic - not a literal translation,
# a natural way to ask the same absent thing.
_UNANSWERABLE_EN = [
    "Does the company offer a pension matching program?",
    "What is the policy on stock options for new hires?",
    "Is there a company policy on sabbatical leave after ten years of service?",
    "Does the company provide childcare subsidies for employees with young children?",
    "What is the policy on gender transition support and related leave?",
    "Is there an internal employee referral bonus program?",
    "What is the policy on keeping pet dinosaurs at the office?",
    "Does the company issue quantum computers to employees?",
    "What is the policy on using personal drones for work tasks?",
    "Is there a company-funded virtual reality headset program?",
    "What is the policy on AI companion robots at employee desks?",
    "Does the company provide a budget for home internet upgrades?",
    "What is the policy on using the company network for cryptocurrency mining?",
    "Is there a policy on holographic meeting rooms?",
    "Can employees claim reimbursement for lottery tickets?",
    "What is the policy on cryptocurrency as a salary payment option?",
    "Does the company reimburse personal yacht rental for client entertainment?",
    "Is there a tax deduction policy for employee pets?",
    "What is the company's policy on stock buyback participation for staff?",
    "Does the company offer interest-free personal loans to employees?",
    "What is the policy on reimbursing personal space travel expenses?",
    "Is there a policy on keeping horses in the staff parking area?",
    "Does the office have an indoor swimming pool booking system?",
    "What is the policy on rooftop helicopter landing pad reservations?",
    "Is there a policy on bringing personal zoo animals into meeting rooms?",
    "Does the company provide on-site overnight sleeping pods?",
    "What is the policy on installing a snow machine in the lobby?",
    "Is there a dedicated napping room policy?",
    "What is the company's policy on drafting contracts with extraterrestrial entities?",
    "Is there a policy on trademarking intergalactic brand names?",
    "What is the policy on hiring artificial intelligence as legal counsel?",
    "Does the company have a policy on time-travel-related contract clauses?",
    "What is the policy on non-disclosure agreements for alien contact?",
    "Is there a policy on cryptocurrency-based legal settlements?",
    "What is the company's stance on patenting perpetual motion machines?",
    "What is the procedure for a client escalation via teleportation?",
    "Is there an incident management procedure for a moon base outage?",
    "What is the policy on using flying saucers for client site visits?",
    "Does the company have a business continuity plan for a zombie outbreak?",
    "What is the on-call policy for underwater data centres?",
    "Is there a service level policy for interplanetary deliveries?",
    "What is the change management policy for quantum computing systems?",
    "What is the policy on advertising campaigns using telepathy?",
    "Is there a policy on hiring mermaids as brand ambassadors?",
    "What is the approval process for dragon-themed marketing campaigns?",
    "Does the company have a policy on holographic billboards?",
    "What is the policy on sponsoring intergalactic sporting events?",
    "Is there a social media policy for posting on Mars colonies?",
    "What is the press policy for announcing time-travel product launches?",
    "What is the procurement policy for purchasing a private island?",
    "Is there a policy on procuring magic beans from suppliers?",
    "What is the approval process for buying a dragon as a company mascot?",
    "Does the company have a supplier policy for unicorn breeders?",
    "What is the procurement threshold for purchasing a small aircraft?",
    "Is there a vendor onboarding process for extraterrestrial suppliers?",
    "What is the company's five-year strategic plan?",
    "Who is the current CEO's favourite football team?",
    "What is the weather forecast for next week at Head Office?",
]

_UNANSWERABLE_AR = [
    "هل تقدم الشركة برنامج مطابقة لمعاش التقاعد؟",
    "ما هي سياسة خيارات الأسهم للموظفين الجدد؟",
    "هل توجد سياسة للإجازة الدراسية بعد عشر سنوات من الخدمة؟",
    "هل تقدم الشركة إعانة لرعاية الأطفال للموظفين؟",
    "ما هي سياسة دعم التحول الجندري والإجازة المرتبطة به؟",
    "هل يوجد برنامج مكافأة داخلي لترشيح الموظفين؟",
    "ما هي سياسة اقتناء الديناصورات الأليفة في المكتب؟",
    "هل تزود الشركة الموظفين بحواسيب كمّية؟",
    "ما هي سياسة استخدام الطائرات المسيّرة الشخصية في العمل؟",
    "هل يوجد برنامج ممول من الشركة لنظارات الواقع الافتراضي؟",
    "ما هي سياسة استخدام الروبوتات المرافقة على مكاتب الموظفين؟",
    "هل تقدم الشركة ميزانية لترقية الإنترنت المنزلي؟",
    "ما هي سياسة استخدام شبكة الشركة لتعدين العملات الرقمية؟",
    "هل توجد سياسة لغرف الاجتماعات الهولوغرافية؟",
    "هل يمكن للموظفين المطالبة باسترداد تكلفة تذاكر اليانصيب؟",
    "ما هي سياسة صرف الراتب بالعملات الرقمية؟",
    "هل تسترد الشركة تكلفة استئجار يخت شخصي لاستضافة العملاء؟",
    "هل توجد سياسة خصم ضريبي لحيوانات الموظفين الأليفة؟",
    "ما هي سياسة مشاركة الموظفين في إعادة شراء أسهم الشركة؟",
    "هل تقدم الشركة قروضًا شخصية بدون فوائد للموظفين؟",
    "ما هي سياسة استرداد نفقات السفر إلى الفضاء الشخصي؟",
    "هل توجد سياسة لإبقاء الخيول في موقف السيارات؟",
    "هل يوجد نظام حجز لحمام سباحة داخلي في المكتب؟",
    "ما هي سياسة حجز منصة هبوط المروحيات على السطح؟",
    "هل توجد سياسة لإحضار حيوانات حديقة شخصية إلى غرف الاجتماعات؟",
    "هل تقدم الشركة كبسولات نوم للمبيت في الموقع؟",
    "ما هي سياسة تركيب آلة ثلج في اللوبي؟",
    "هل توجد سياسة مخصصة لغرفة القيلولة؟",
    "ما هي سياسة الشركة بشأن صياغة عقود مع كيانات خارج كوكب الأرض؟",
    "هل توجد سياسة لتسجيل العلامات التجارية بين المجرات؟",
    "ما هي سياسة توظيف الذكاء الاصطناعي كمستشار قانوني؟",
    "هل توجد سياسة لبنود العقود المتعلقة بالسفر عبر الزمن؟",
    "ما هي سياسة اتفاقيات عدم الإفصاح الخاصة بالتواصل مع الكائنات الفضائية؟",
    "هل توجد سياسة لتسويات قانونية باستخدام العملات الرقمية؟",
    "ما هو موقف الشركة من تسجيل براءات اختراع لآلات الحركة الدائمة؟",
    "ما هو إجراء تصعيد شكاوى العملاء عبر الانتقال الآني؟",
    "هل يوجد إجراء لإدارة الحوادث في قاعدة على القمر؟",
    "ما هي سياسة استخدام الصحون الطائرة لزيارات مواقع العملاء؟",
    "هل لدى الشركة خطة استمرارية أعمال لتفشي الزومبي؟",
    "ما هي سياسة المناوبة لمراكز بيانات تحت الماء؟",
    "هل توجد سياسة مستوى خدمة للتوصيل بين الكواكب؟",
    "ما هي سياسة إدارة التغيير لأنظمة الحوسبة الكمّية؟",
    "ما هي سياسة الحملات الإعلانية باستخدام التخاطر؟",
    "هل توجد سياسة لتوظيف حوريات البحر كسفيرات للعلامة التجارية؟",
    "ما هي إجراءات الموافقة على الحملات التسويقية بطابع التنانين؟",
    "هل لدى الشركة سياسة للوحات الإعلانية الهولوغرافية؟",
    "ما هي سياسة رعاية الفعاليات الرياضية بين المجرات؟",
    "هل توجد سياسة تواصل اجتماعي للنشر من مستعمرات المريخ؟",
    "ما هي سياسة الصحافة للإعلان عن منتجات السفر عبر الزمن؟",
    "ما هي سياسة المشتريات لشراء جزيرة خاصة؟",
    "هل توجد سياسة لشراء حبوب سحرية من الموردين؟",
    "ما هي إجراءات الموافقة على شراء تنين كتميمة للشركة؟",
    "هل لدى الشركة سياسة موردين لمربي وحيد القرن؟",
    "ما هو حد المشتريات المطلوب للموافقة على شراء طائرة صغيرة؟",
    "هل يوجد إجراء تأهيل موردين للموردين من خارج كوكب الأرض؟",
    "ما هي الخطة الاستراتيجية للشركة على مدى خمس سنوات؟",
    "ما هو فريق كرة القدم المفضل للرئيس التنفيذي الحالي؟",
    "ما هي حالة الطقس المتوقعة الأسبوع المقبل في المكتب الرئيسي؟",
]

assert len(_UNANSWERABLE_EN) == len(_UNANSWERABLE_AR), "EN/AR unanswerable lists must line up"


def _load_topic_docs() -> tuple[dict[tuple[str, str], str], dict[str, list[str]]]:
    """Two lookups, both restricted to Head Office documents:

    - faq_doc_id[(topic_key, lang)]: the FAQ document that a given
      question's text actually came from.
    - family_docs[topic_key]: every document for that topic, Policy through
      Quick Reference, in BOTH languages. Retrieval is cross-lingual by
      design (Section 9) - a question in one language can legitimately be
      answered from the other language's document - so citing any of these
      is a correct, sourced answer, not only the one the question came from.
    """
    faq_doc_id: dict[tuple[str, str], str] = {}
    family_docs: dict[str, list[str]] = {}
    with (KB_DIR / "manifest.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["office"] not in _OFFICE_NAME.values():
                continue
            family_docs.setdefault(row["topic_key"], []).append(row["doc_id"])
            if row["type"] == "FAQ":
                faq_doc_id[(row["topic_key"], row["language"])] = row["doc_id"]
    return faq_doc_id, family_docs


def _faq_items() -> list[dict]:
    """Family A and B: every FAQ question, real and reworded."""
    faq_doc_id, family_docs = _load_topic_docs()
    items: list[dict] = []
    counter = 0

    for topic in gen.TOPICS:
        for lang in ("en", "ar"):
            doc_id = faq_doc_id.get((topic.key, lang))
            docs = family_docs.get(topic.key)
            if not doc_id or not docs:
                continue  # this topic's documents didn't make the 1000-document slice

            # Reproduce the exact values used when this document's facts were
            # built (shared across doc types AND languages - see
            # generate_kb.py's facts_rng).
            rng = random.Random(f"{gen.SEED}:{topic.key}:HO:facts")
            values = gen.pick_params(topic, rng, 0)
            values["owner"] = gen.owner_name(topic.dept, lang)

            for faq_index, (q_en, q_ar, a_en, a_ar) in enumerate(topic.faq):
                q = gen.fill(q_en if lang == "en" else q_ar, values)
                a = gen.fill(a_en if lang == "en" else a_ar, values)
                counter += 1
                items.append(
                    {
                        "id": f"A{counter:04d}",
                        "family": "faq_exact",
                        "question": q,
                        "language": lang,
                        "department": topic.dept,
                        "topic": topic.key,
                        "expected_doc": doc_id,
                        "expected_docs": docs,
                        "expected_fact": a,
                        "answerable": True,
                    }
                )
                items.append(
                    {
                        "id": f"B{counter:04d}",
                        "family": "faq_paraphrased",
                        "question": _paraphrase(q, lang, faq_index),
                        "language": lang,
                        "department": topic.dept,
                        "topic": topic.key,
                        "expected_doc": doc_id,
                        "expected_docs": docs,
                        "expected_fact": a,
                        "answerable": True,
                    }
                )
    return items


def _unanswerable_items() -> list[dict]:
    items = []
    for index, (q_en, q_ar) in enumerate(zip(_UNANSWERABLE_EN, _UNANSWERABLE_AR, strict=True)):
        for lang, question in (("en", q_en), ("ar", q_ar)):
            items.append(
                {
                    "id": f"C{index:04d}{lang}",
                    "family": "unanswerable",
                    "question": question,
                    "language": lang,
                    "department": None,
                    "topic": None,
                    "expected_doc": None,
                    "expected_docs": [],
                    "expected_fact": None,
                    "answerable": False,
                }
            )
    return items


def main() -> None:
    items = _faq_items() + _unanswerable_items()
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    answerable = sum(1 for i in items if i["answerable"])
    half = answerable // 2
    print(f"wrote {len(items)} questions to {OUT_PATH}")
    print(f"  answerable   : {answerable}  ({half} exact + {half} paraphrased)")
    print(f"  unanswerable : {len(items) - answerable}")
    langs = {i["language"] for i in items}
    for lang in sorted(langs):
        print(f"  {lang}: {sum(1 for i in items if i['language'] == lang)}")


if __name__ == "__main__":
    main()
