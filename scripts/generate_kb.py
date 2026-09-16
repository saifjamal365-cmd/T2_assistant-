"""
generate_kb.py
==============
Builds the synthetic knowledge base for the T2 Assistant.

It writes ~2000 company-policy documents (1000 English + 1000 Arabic) under
`data/knowledge_base/`, plus `manifest.csv` describing every file.

The corpus is:
  * built from ~114 hand-written topics across 8 departments,
  * expanded with per-office variants (numbers and scope differ),
  * seeded, so re-running produces the exact same corpus.

Run:  python scripts/generate_kb.py           (default: 2000 documents)
      python scripts/generate_kb.py --count 500 --out data/kb2

Every document has a header (ID, department, type, version, effective date,
status, owner, applies-to, related documents) and a body. Some documents are
marked "Superseded" and point to the version that replaced them, so the
retrieval layer has to deal with old versions - this is on purpose.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Static configuration
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "knowledge_base"
SEED = 20260909
ORG = "T2"
CONTACT_DOMAIN = "t2.example"

DEPARTMENTS = {
    "HR": "Human Resources",
    "IT": "Information Technology",
    "FIN": "Finance",
    "FAC": "Facilities",
    "LEG": "Legal & Compliance",
    "OPS": "Operations",
    "MKT": "Marketing & Communications",
    "PRC": "Procurement",
}

# offices: (key, English name, Arabic name, headcount weight, param shift)
OFFICES = [
    ("HO", "Head Office", "المكتب الرئيسي", 1.0, 0),
    ("RUH", "Riyadh Office", "مكتب الرياض", 0.9, 0),
    ("JED", "Jeddah Office", "مكتب جدة", 0.6, -1),
    ("DXB", "Dubai Office", "مكتب دبي", 0.5, +1),
    ("CAI", "Cairo Office", "مكتب القاهرة", 0.7, -1),
    ("REM", "Remote-first Teams", "الفرق العاملة عن بُعد", 0.4, +1),
]

DOC_TYPES = ["Policy", "Procedure", "FAQ", "Quick Reference"]

OWNERS = {
    "HR": ("HR Operations", "عمليات الموارد البشرية"),
    "IT": ("IT Service Desk", "مكتب خدمات تقنية المعلومات"),
    "FIN": ("Finance Shared Services", "الخدمات المشتركة للمالية"),
    "FAC": ("Facilities Management", "إدارة المرافق"),
    "LEG": ("Legal & Compliance Office", "مكتب الشؤون القانونية والامتثال"),
    "OPS": ("Operations Excellence", "التميز التشغيلي"),
    "MKT": ("Corporate Communications", "الاتصال المؤسسي"),
    "PRC": ("Procurement Office", "مكتب المشتريات"),
}

# --------------------------------------------------------------------------
# Topic bank
# --------------------------------------------------------------------------


@dataclass
class Topic:
    key: str
    dept: str
    title_en: str
    title_ar: str
    purpose_en: str
    purpose_ar: str
    clauses: list[tuple[str, str]]  # (english, arabic), may contain {params}
    params: dict[str, list] = field(default_factory=dict)
    faq: list[tuple[str, str, str, str]] = field(default_factory=list)  # q_en, q_ar, a_en, a_ar
    related: list[str] = field(default_factory=list)
    office_specific: bool = True


def T(*a, **kw) -> Topic:
    return Topic(*a, **kw)


TOPICS: list[Topic] = [
    # ---------------- HR ----------------
    T("annual_leave", "HR",
      "Annual Leave Policy", "سياسة الإجازة السنوية",
      "This policy sets out how much paid annual leave employees receive and how to request it.",
      "تحدد هذه السياسة عدد أيام الإجازة السنوية المدفوعة التي يحصل عليها الموظفون وكيفية طلبها.",
      [("Full-time employees are entitled to {leave_days} paid annual leave days per year, accrued monthly.",
        "يستحق الموظفون بدوام كامل {leave_days} يومًا من الإجازة السنوية المدفوعة سنويًا، تُحتسب شهريًا."),
       ("Requests must be submitted at least {notice} working days in advance through the HR system.",
        "يجب تقديم الطلبات قبل {notice} أيام عمل على الأقل عبر نظام الموارد البشرية."),
       ("Up to {carryover} unused days may be carried into the next year; any excess is forfeited.",
        "يمكن ترحيل ما يصل إلى {carryover} أيام غير مستخدمة إلى العام التالي، ويُلغى ما يزيد على ذلك."),
       ("Leave cannot be combined with sick leave for the same period.",
        "لا يمكن دمج الإجازة السنوية مع الإجازة المرضية في الفترة نفسها.")],
      {"leave_days": [21, 24, 25, 28, 30], "notice": [3, 5, 7], "carryover": [5, 7, 10]},
      [("How many annual leave days do I get?", "كم يوم إجازة سنوية أحصل عليه؟",
        "Full-time employees get {leave_days} paid days per year.",
        "يحصل الموظفون بدوام كامل على {leave_days} يومًا مدفوعًا سنويًا."),
       ("How far in advance must I request leave?", "قبل كم من الوقت يجب طلب الإجازة؟",
        "At least {notice} working days before the leave starts.",
        "قبل {notice} أيام عمل على الأقل من بدء الإجازة."),
       ("Can I carry unused days to next year?", "هل يمكن ترحيل الأيام غير المستخدمة؟",
        "Yes, up to {carryover} days; the rest is lost.",
        "نعم، حتى {carryover} أيام، ويُفقد الباقي.")],
      ["sick_leave", "parental_leave", "unpaid_leave"]),

    T("sick_leave", "HR",
      "Sick Leave Policy", "سياسة الإجازة المرضية",
      "This policy explains sick-leave entitlement and when a medical report is needed.",
      "توضح هذه السياسة استحقاق الإجازة المرضية ومتى يلزم تقديم تقرير طبي.",
      [("Employees may take up to {sick_days} paid sick days per year.",
        "يجوز للموظف أخذ ما يصل إلى {sick_days} يوم إجازة مرضية مدفوعة سنويًا."),
       ("A medical report is required for any sick leave longer than {report_after} consecutive days.",
        "يلزم تقرير طبي لأي إجازة مرضية تزيد عن {report_after} أيام متتالية."),
       ("The employee must notify their manager on the first day of absence.",
        "يجب على الموظف إبلاغ مديره في اليوم الأول للغياب."),
       ("Unused sick days do not carry over and are not paid out.",
        "لا تُرحَّل أيام الإجازة المرضية غير المستخدمة ولا تُصرف نقدًا.")],
      {"sick_days": [10, 12, 15, 20], "report_after": [2, 3, 5]},
      [("Do I need a doctor's note for two sick days?", "هل أحتاج تقريرًا طبيًا ليومين مرضيين؟",
        "No. A report is only needed after {report_after} consecutive days.",
        "لا. يلزم التقرير فقط بعد {report_after} أيام متتالية."),
       ("How many paid sick days per year?", "كم يوم إجازة مرضية مدفوعة سنويًا؟",
        "Up to {sick_days} days.", "حتى {sick_days} يومًا.")],
      ["annual_leave", "unpaid_leave", "workspace_ergonomics"]),

    T("parental_leave", "HR",
      "Parental Leave Policy", "سياسة إجازة الوالدية",
      "This policy covers paid leave for the birth or adoption of a child.",
      "تغطي هذه السياسة الإجازة المدفوعة عند ولادة طفل أو تبنيه.",
      [("The primary caregiver receives {primary} paid days; the secondary caregiver receives {secondary} paid days.",
        "يحصل مقدم الرعاية الأساسي على {primary} يومًا مدفوعًا، ومقدم الرعاية الثانوي على {secondary} يومًا مدفوعًا."),
       ("Requests should reach HR at least {notice} days before the expected start date with supporting documents.",
        "يجب أن تصل الطلبات إلى الموارد البشرية قبل {notice} يومًا على الأقل من التاريخ المتوقع مع المستندات الداعمة."),
       ("Leave may be split into two blocks within the first year after the birth or adoption.",
        "يمكن تقسيم الإجازة إلى فترتين خلال السنة الأولى بعد الولادة أو التبني."),
       ("Job protection applies: the employee returns to the same or an equivalent role.",
        "تُطبَّق حماية الوظيفة: يعود الموظف إلى الدور نفسه أو دور مكافئ.")],
      {"primary": [45, 60, 70, 90], "secondary": [5, 10, 14], "notice": [15, 30]},
      [("How much parental leave for the primary caregiver?", "كم مدة إجازة الوالدية لمقدم الرعاية الأساسي؟",
        "{primary} paid days.", "{primary} يومًا مدفوعًا."),
       ("Can the leave be split?", "هل يمكن تقسيم الإجازة؟",
        "Yes, into two blocks in the first year.", "نعم، إلى فترتين خلال السنة الأولى.")],
      ["annual_leave", "sick_leave", "relocation"]),

    T("bereavement_leave", "HR",
      "Bereavement Leave Policy", "سياسة إجازة الوفاة",
      "This policy sets paid leave after the death of a close family member.",
      "تحدد هذه السياسة الإجازة المدفوعة بعد وفاة أحد أفراد الأسرة المقربين.",
      [("Employees receive {days} paid days on the death of a spouse, parent, child or sibling.",
        "يحصل الموظفون على {days} أيام مدفوعة عند وفاة الزوج أو أحد الوالدين أو الأبناء أو الإخوة."),
       ("For other relatives, {other_days} paid day(s) are granted.",
        "بالنسبة لبقية الأقارب، يُمنح {other_days} يوم مدفوع."),
       ("The manager may approve additional unpaid days on request.",
        "يجوز للمدير الموافقة على أيام إضافية غير مدفوعة عند الطلب.")],
      {"days": [3, 5, 7], "other_days": [1, 2]},
      [("How many days for the loss of a parent?", "كم يومًا عند وفاة أحد الوالدين؟",
        "{days} paid days.", "{days} أيام مدفوعة.")],
      ["annual_leave", "grievance"]),

    T("unpaid_leave", "HR",
      "Unpaid Leave Policy", "سياسة الإجازة غير المدفوعة",
      "This policy explains when employees can take unpaid leave.",
      "توضح هذه السياسة متى يمكن للموظفين أخذ إجازة غير مدفوعة.",
      [("Unpaid leave of up to {max_months} months may be requested after paid leave is exhausted.",
        "يمكن طلب إجازة غير مدفوعة تصل إلى {max_months} أشهر بعد استنفاد الإجازة المدفوعة."),
       ("Approval requires the line manager and the department head to sign off.",
        "تتطلب الموافقة توقيع المدير المباشر ورئيس القسم."),
       ("Benefits are paused during unpaid leave longer than {pause_after} days.",
        "تُعلَّق المزايا خلال الإجازة غير المدفوعة التي تزيد عن {pause_after} يومًا.")],
      {"max_months": [3, 6, 12], "pause_after": [30, 60]},
      [("How long can unpaid leave be?", "ما أقصى مدة للإجازة غير المدفوعة؟",
        "Up to {max_months} months.", "حتى {max_months} أشهر.")],
      ["annual_leave", "sick_leave"]),

    T("working_hours", "HR",
      "Working Hours Policy", "سياسة ساعات العمل",
      "This policy defines standard working hours and core hours.",
      "تحدد هذه السياسة ساعات العمل القياسية والساعات الأساسية.",
      [("The standard work week is {hours} hours over {days} days.",
        "أسبوع العمل القياسي هو {hours} ساعة على مدى {days} أيام."),
       ("Core hours, when everyone must be reachable, are {core_start} to {core_end}.",
        "الساعات الأساسية التي يجب أن يكون الجميع متاحين فيها هي من {core_start} إلى {core_end}."),
       ("Time is recorded in the HR system; managers review timesheets weekly.",
        "يُسجَّل الوقت في نظام الموارد البشرية، ويراجع المديرون كشوف الدوام أسبوعيًا.")],
      {"hours": [40, 42, 45], "days": [5], "core_start": ["10:00"], "core_end": ["15:00", "16:00"]},
      [("What are the core hours?", "ما هي الساعات الأساسية؟",
        "{core_start} to {core_end}.", "من {core_start} إلى {core_end}.")],
      ["overtime", "flexible_hours", "remote_work"]),

    T("overtime", "HR",
      "Overtime Policy", "سياسة العمل الإضافي",
      "This policy explains when overtime is paid and how it is approved.",
      "توضح هذه السياسة متى يُدفع العمل الإضافي وكيف تتم الموافقة عليه.",
      [("Overtime must be approved in writing by the manager before it is worked.",
        "يجب اعتماد العمل الإضافي كتابيًا من المدير قبل أدائه."),
       ("Overtime is paid at {rate} times the normal hourly rate.",
        "يُدفع العمل الإضافي بمعدل {rate} أضعاف الأجر الساعي المعتاد."),
       ("Employees at grade {exempt_grade} and above are not eligible for overtime pay.",
        "الموظفون في الدرجة {exempt_grade} فأعلى غير مؤهلين لأجر العمل الإضافي.")],
      {"rate": ["1.25", "1.5", "2"], "exempt_grade": ["M3", "M4", "D1"]},
      [("Is overtime paid without approval?", "هل يُدفع العمل الإضافي بدون موافقة؟",
        "No. It must be approved in writing first.", "لا. يجب اعتماده كتابيًا أولًا.")],
      ["working_hours", "on_call"]),

    T("remote_work", "HR",
      "Remote Work Policy", "سياسة العمل عن بُعد",
      "This policy covers working from home and fully remote arrangements.",
      "تغطي هذه السياسة العمل من المنزل والترتيبات البعيدة بالكامل.",
      [("Approved employees may work remotely up to {remote_days} days per week.",
        "يجوز للموظفين المعتمدين العمل عن بُعد حتى {remote_days} أيام في الأسبوع."),
       ("Remote employees must join scheduled meetings with the camera on unless agreed otherwise.",
        "يجب على الموظفين عن بُعد حضور الاجتماعات المجدولة مع تشغيل الكاميرا ما لم يُتفق على غير ذلك."),
       ("Company equipment used remotely must connect only through the approved VPN.",
        "يجب أن تتصل معدات الشركة المستخدمة عن بُعد عبر شبكة VPN المعتمدة فقط."),
       ("A fully remote arrangement needs HR approval and is reviewed every {review_months} months.",
        "يتطلب الترتيب البعيد بالكامل موافقة الموارد البشرية ويُراجع كل {review_months} أشهر.")],
      {"remote_days": [2, 3, 4], "review_months": [6, 12]},
      [("How many days can I work from home?", "كم يومًا يمكنني العمل من المنزل؟",
        "Up to {remote_days} days per week.", "حتى {remote_days} أيام في الأسبوع."),
       ("Do I need the VPN when working remotely?", "هل أحتاج VPN عند العمل عن بُعد؟",
        "Yes, company equipment must use the approved VPN.",
        "نعم، يجب أن تستخدم معدات الشركة شبكة VPN المعتمدة.")],
      ["working_hours", "vpn_remote_access", "hardware_request"]),

    T("flexible_hours", "HR",
      "Flexible Hours Policy", "سياسة ساعات العمل المرنة",
      "This policy lets employees shift their start and end times within limits.",
      "تتيح هذه السياسة للموظفين تعديل أوقات البدء والانتهاء ضمن حدود.",
      [("Start time may be chosen between {early} and {late}, keeping the full daily hours.",
        "يمكن اختيار وقت البدء بين {early} و{late} مع الحفاظ على ساعات اليوم كاملة."),
       ("Any flexible arrangement must still cover the core hours.",
        "يجب أن يغطي أي ترتيب مرن الساعات الأساسية.")],
      {"early": ["07:00", "07:30"], "late": ["09:30", "10:00"]},
      [("Can I start at 9:30?", "هل يمكنني البدء الساعة 9:30؟",
        "Yes, if you still cover the core hours and full daily hours.",
        "نعم، إذا غطيت الساعات الأساسية وساعات اليوم كاملة.")],
      ["working_hours", "remote_work"]),

    T("probation", "HR",
      "Probation Policy", "سياسة فترة التجربة",
      "This policy sets the probation period for new hires and how it is assessed.",
      "تحدد هذه السياسة فترة التجربة للموظفين الجدد وكيفية تقييمها.",
      [("The probation period is {months} months from the start date.",
        "فترة التجربة {months} أشهر من تاريخ المباشرة."),
       ("A review meeting is held at the midpoint and at the end of probation.",
        "يُعقد اجتماع مراجعة في منتصف الفترة وفي نهايتها."),
       ("During probation, either side may end the contract with {notice} days' notice.",
        "خلال فترة التجربة، يجوز لأي طرف إنهاء العقد بإشعار مدته {notice} أيام.")],
      {"months": [3, 6], "notice": [7, 14]},
      [("How long is probation?", "كم مدة فترة التجربة؟",
        "{months} months.", "{months} أشهر.")],
      ["onboarding", "performance_review", "resignation_notice"]),

    T("performance_review", "HR",
      "Performance Review Policy", "سياسة تقييم الأداء",
      "This policy explains how and when performance is reviewed.",
      "توضح هذه السياسة كيف ومتى يُراجَع الأداء.",
      [("Formal reviews take place {cycles} times a year, in {months_list}.",
        "تُجرى المراجعات الرسمية {cycles} مرات في السنة، في {months_list}."),
       ("Each review includes a self-assessment, manager feedback and, for senior roles, peer input.",
        "تتضمن كل مراجعة تقييمًا ذاتيًا وملاحظات المدير، وللأدوار العليا آراء الزملاء."),
       ("A rating of \"needs improvement\" for two cycles in a row triggers a performance plan.",
        "يؤدي تقييم \"يحتاج إلى تحسين\" لدورتين متتاليتين إلى وضع خطة أداء.")],
      {"cycles": [1, 2], "months_list": ["June and December", "March, June, September and December"]},
      [("How often are performance reviews?", "كم مرة يُراجَع الأداء؟",
        "{cycles} times a year.", "{cycles} مرات في السنة.")],
      ["probation", "promotion", "salary_review"]),

    T("promotion", "HR",
      "Promotion Policy", "سياسة الترقية",
      "This policy describes how promotions are proposed and approved.",
      "تصف هذه السياسة كيفية اقتراح الترقيات والموافقة عليها.",
      [("Promotions are considered {cycles} time(s) a year, aligned with the review cycle.",
        "يُنظر في الترقيات {cycles} مرة في السنة، بما يتماشى مع دورة التقييم."),
       ("An employee must have at least {min_months} months in role and a rating of \"meets\" or above.",
        "يجب أن يكون للموظف {min_months} أشهر على الأقل في الدور وتقييم \"يفي\" أو أعلى."),
       ("Promotions require the department head and HR to approve.",
        "تتطلب الترقيات موافقة رئيس القسم والموارد البشرية.")],
      {"cycles": [1, 2], "min_months": [12, 18, 24]},
      [("How long before I can be promoted?", "كم يجب أن أمضي قبل الترقية؟",
        "At least {min_months} months in role.", "{min_months} أشهر على الأقل في الدور.")],
      ["performance_review", "salary_review"]),

    T("salary_review", "HR",
      "Salary Review Policy", "سياسة مراجعة الرواتب",
      "This policy explains the annual salary review.",
      "توضح هذه السياسة مراجعة الرواتب السنوية.",
      [("Salaries are reviewed once a year, effective {effective_month}.",
        "تُراجَع الرواتب مرة واحدة سنويًا، وتسري اعتبارًا من {effective_month}."),
       ("Adjustments depend on the performance rating, the salary band and the annual budget.",
        "تعتمد التعديلات على تقييم الأداء ونطاق الراتب والميزانية السنوية."),
       ("Out-of-cycle adjustments need Compensation Committee approval.",
        "تتطلب التعديلات خارج الدورة موافقة لجنة التعويضات.")],
      {"effective_month": ["1 January", "1 April", "1 July"]},
      [("When do salary changes take effect?", "متى تسري تغييرات الراتب؟",
        "From {effective_month}.", "اعتبارًا من {effective_month}.")],
      ["performance_review", "promotion"]),

    T("onboarding", "HR",
      "Onboarding Policy", "سياسة الإلحاق الوظيفي",
      "This policy lists what must happen when a new employee joins.",
      "تسرد هذه السياسة ما يجب أن يحدث عند انضمام موظف جديد.",
      [("Before day one, HR sends a welcome email with the start date, location and documents to bring.",
        "قبل اليوم الأول، ترسل الموارد البشرية بريدًا ترحيبيًا يتضمن تاريخ المباشرة والموقع والمستندات المطلوبة."),
       ("On day one, the manager introduces the team and confirms system access is active.",
        "في اليوم الأول، يعرّف المدير بالفريق ويتأكد من تفعيل الوصول إلى الأنظمة."),
       ("Within the first {weeks} weeks, the employee completes mandatory training on conduct, privacy and security.",
        "خلال أول {weeks} أسابيع، يُكمل الموظف التدريب الإلزامي على السلوك والخصوصية والأمن.")],
      {"weeks": [1, 2, 4]},
      [("What training must a new hire complete?", "ما التدريب الإلزامي للموظف الجديد؟",
        "Conduct, data privacy and security, in the first {weeks} weeks.",
        "السلوك وخصوصية البيانات والأمن، خلال أول {weeks} أسابيع.")],
      ["probation", "code_of_conduct", "system_access"]),

    T("offboarding", "HR",
      "Offboarding Policy", "سياسة إنهاء الخدمة",
      "This policy lists what must happen when an employee leaves.",
      "تسرد هذه السياسة ما يجب أن يحدث عند مغادرة موظف.",
      [("The manager submits the leaver form at least {notice} working days before the last day.",
        "يقدم المدير نموذج المغادرة قبل {notice} أيام عمل على الأقل من آخر يوم."),
       ("On the last day, all equipment and access cards are returned and accounts are disabled.",
        "في اليوم الأخير، تُعاد جميع المعدات وبطاقات الدخول وتُعطَّل الحسابات."),
       ("Final pay, including unused annual leave, is settled within {settle} days.",
        "تُسوَّى المستحقات النهائية، بما فيها الإجازة السنوية غير المستخدمة، خلال {settle} يومًا.")],
      {"notice": [5, 10], "settle": [7, 14, 30]},
      [("When is final pay settled?", "متى تُسوَّى المستحقات النهائية؟",
        "Within {settle} days of the last working day.", "خلال {settle} يومًا من آخر يوم عمل.")],
      ["resignation_notice", "asset_return", "system_access"]),

    T("resignation_notice", "HR",
      "Resignation and Notice Policy", "سياسة الاستقالة ومدة الإشعار",
      "This policy sets the notice period for resignation.",
      "تحدد هذه السياسة مدة الإشعار عند الاستقالة.",
      [("Employees give {notice_weeks} weeks' written notice; grade {senior_grade} and above give {senior_weeks} weeks.",
        "يقدم الموظفون إشعارًا كتابيًا مدته {notice_weeks} أسابيع، ومن الدرجة {senior_grade} فأعلى {senior_weeks} أسابيع."),
       ("Notice may be shortened only with the manager's written agreement.",
        "لا يجوز تقصير مدة الإشعار إلا بموافقة كتابية من المدير."),
       ("An exit interview is offered before the last day.",
        "تُعرض مقابلة إنهاء الخدمة قبل اليوم الأخير.")],
      {"notice_weeks": [2, 4], "senior_grade": ["M3", "M4"], "senior_weeks": [8, 12]},
      [("What is the notice period?", "ما مدة الإشعار؟",
        "{notice_weeks} weeks for most roles.", "{notice_weeks} أسابيع لمعظم الأدوار.")],
      ["offboarding", "references"]),

    T("references", "HR",
      "Employment References Policy", "سياسة توصيات التوظيف",
      "This policy states how the company responds to reference requests.",
      "تحدد هذه السياسة كيفية رد الشركة على طلبات التوصية.",
      [("Only HR may issue an official reference; it confirms title and dates of employment only.",
        "الموارد البشرية فقط من يصدر توصية رسمية، وتؤكد المسمى الوظيفي وتواريخ العمل فقط."),
       ("Personal references from managers must state that they are personal, not on behalf of {org}.",
        "يجب أن توضح التوصيات الشخصية من المديرين أنها شخصية وليست باسم {org}.")],
      {},
      [("Will the company give a detailed reference?", "هل تقدم الشركة توصية تفصيلية؟",
        "No. HR confirms title and dates of employment only.",
        "لا. تؤكد الموارد البشرية المسمى وتواريخ العمل فقط.")],
      ["offboarding", "confidentiality"], office_specific=False),

    T("code_of_conduct", "HR",
      "Code of Conduct", "مدونة السلوك",
      "The code sets the standard of behaviour expected from every employee.",
      "تحدد المدونة معيار السلوك المتوقع من كل موظف.",
      [("Treat colleagues and clients with respect; discrimination and harassment are not tolerated.",
        "عامل الزملاء والعملاء باحترام، ولا تُقبل ممارسات التمييز أو التحرش."),
       ("Any conflict of interest must be disclosed in writing to {owner}.",
        "يجب الإفصاح عن أي تضارب مصالح كتابيًا إلى {owner}."),
       ("Breaching the code may lead to disciplinary action up to termination.",
        "قد يؤدي خرق المدونة إلى إجراء تأديبي قد يصل إلى إنهاء الخدمة.")],
      {},
      [("Where do I report a conflict of interest?", "أين أُبلغ عن تضارب المصالح؟",
        "In writing to {owner}.", "كتابيًا إلى {owner}.")],
      ["anti_harassment", "conflict_of_interest", "gifts_hospitality"], office_specific=False),

    T("anti_harassment", "HR",
      "Anti-Harassment Policy", "سياسة مكافحة التحرش",
      "This policy prohibits harassment and explains how to report it.",
      "تحظر هذه السياسة التحرش وتوضح كيفية الإبلاغ عنه.",
      [("Harassment of any kind is prohibited and is treated as serious misconduct.",
        "يُحظر التحرش بجميع أشكاله ويُعامل بوصفه سوء سلوك جسيم."),
       ("Reports can be made to the manager, HR, or the confidential channel, and may be anonymous.",
        "يمكن الإبلاغ إلى المدير أو الموارد البشرية أو القناة السرية، ويجوز أن يكون البلاغ مجهولًا."),
       ("Retaliation against someone who reports in good faith is itself a violation.",
        "الانتقام ممن يبلغ بحسن نية يُعد بحد ذاته مخالفة.")],
      {},
      [("Can I report harassment anonymously?", "هل يمكنني الإبلاغ عن التحرش بشكل مجهول؟",
        "Yes, through the confidential channel.", "نعم، عبر القناة السرية.")],
      ["code_of_conduct", "grievance", "whistleblowing"], office_specific=False),

    T("grievance", "HR",
      "Grievance Policy", "سياسة التظلمات",
      "This policy explains how to raise a formal complaint at work.",
      "توضح هذه السياسة كيفية تقديم شكوى رسمية في العمل.",
      [("Raise the issue informally with the manager first where possible.",
        "أثر المسألة بشكل غير رسمي مع المدير أولًا حيثما أمكن."),
       ("A formal grievance is submitted in writing and acknowledged within {ack} working days.",
        "يُقدَّم التظلم الرسمي كتابيًا ويُقر باستلامه خلال {ack} أيام عمل."),
       ("A hearing is held within {hearing} working days, and the outcome is given in writing.",
        "تُعقد جلسة خلال {hearing} أيام عمل، وتُبلَّغ النتيجة كتابيًا.")],
      {"ack": [3, 5], "hearing": [10, 15]},
      [("How quickly is a grievance acknowledged?", "متى يُقر باستلام التظلم؟",
        "Within {ack} working days.", "خلال {ack} أيام عمل.")],
      ["anti_harassment", "code_of_conduct"], office_specific=False),

    T("dress_code", "HR",
      "Dress Code Policy", "سياسة الزي في العمل",
      "This policy describes what to wear in the office and with clients.",
      "تصف هذه السياسة ما يُرتدى في المكتب ومع العملاء.",
      [("Standard office wear is smart-casual.",
        "الزي المكتبي القياسي هو الكاجوال الأنيق."),
       ("Client meetings and external events require business attire.",
        "تتطلب اجتماعات العملاء والفعاليات الخارجية الزي الرسمي."),
       ("{casual_day} is a casual day, except where a client is on site.",
        "{casual_day} يوم كاجوال، إلا عند وجود عميل في الموقع.")],
      {"casual_day": ["Thursday", "Friday"]},
      [("Is there a casual day?", "هل يوجد يوم كاجوال؟",
        "Yes, {casual_day}, unless a client is on site.",
        "نعم، {casual_day}، إلا عند وجود عميل في الموقع.")],
      ["visitor_management"]),

    T("training_development", "HR",
      "Training and Development Policy", "سياسة التدريب والتطوير",
      "This policy covers the training budget and study support.",
      "تغطي هذه السياسة ميزانية التدريب ودعم الدراسة.",
      [("Each employee has an annual training budget of {budget} {currency}.",
        "لكل موظف ميزانية تدريب سنوية قدرها {budget} {currency}."),
       ("Courses must be approved by the manager and be relevant to the current or next role.",
        "يجب اعتماد الدورات من المدير وأن تكون ذات صلة بالدور الحالي أو التالي."),
       ("For a certification the company pays for, the employee agrees to stay {retain_months} months or repay pro-rata.",
        "مقابل شهادة تدفع الشركة تكلفتها، يوافق الموظف على البقاء {retain_months} شهرًا أو السداد بالتناسب.")],
      {"budget": [3000, 5000, 8000], "currency": ["SAR", "AED", "EGP"], "retain_months": [6, 12]},
      [("What is my training budget?", "ما ميزانية التدريب الخاصة بي؟",
        "{budget} {currency} per year.", "{budget} {currency} سنويًا.")],
      ["performance_review"]),

    T("secondment", "HR",
      "Secondment Policy", "سياسة الإعارة الداخلية",
      "This policy covers temporary moves to another team or office.",
      "تغطي هذه السياسة الانتقال المؤقت إلى فريق أو مكتب آخر.",
      [("A secondment lasts {min_months} to {max_months} months and needs both managers to agree.",
        "تستمر الإعارة من {min_months} إلى {max_months} أشهر وتتطلب موافقة المديرين."),
       ("The employee returns to their original role or an equivalent one at the end.",
        "يعود الموظف إلى دوره الأصلي أو دور مكافئ عند الانتهاء.")],
      {"min_months": [3, 6], "max_months": [12, 18, 24]},
      [("How long can a secondment last?", "ما أقصى مدة للإعارة؟",
        "Up to {max_months} months.", "حتى {max_months} أشهر.")],
      ["relocation", "promotion"], office_specific=False),

    T("relocation", "HR",
      "Relocation Support Policy", "سياسة دعم الانتقال",
      "This policy explains support when an employee moves offices or cities for work.",
      "توضح هذه السياسة الدعم عند انتقال الموظف بين المكاتب أو المدن للعمل.",
      [("The company covers shipping up to {weight} kg and {nights} nights of temporary accommodation.",
        "تغطي الشركة الشحن حتى {weight} كجم و{nights} ليلة من الإقامة المؤقتة."),
       ("A relocation allowance of {allowance} {currency} is paid once, on arrival.",
        "يُدفع بدل انتقال قدره {allowance} {currency} مرة واحدة عند الوصول."),
       ("If the employee resigns within {clawback_months} months, the allowance is repaid pro-rata.",
        "إذا استقال الموظف خلال {clawback_months} شهرًا، يُسترد البدل بالتناسب.")],
      {"weight": [100, 200, 300], "nights": [7, 14, 30], "allowance": [5000, 10000],
       "currency": ["SAR", "AED", "EGP"], "clawback_months": [12, 24]},
      [("Is there a relocation allowance?", "هل يوجد بدل انتقال؟",
        "Yes, {allowance} {currency}, paid once on arrival.",
        "نعم، {allowance} {currency}، تُدفع مرة واحدة عند الوصول.")],
      ["secondment", "parental_leave"], office_specific=False),

    # ---------------- IT ----------------
    T("password_policy", "IT",
      "Password Policy", "سياسة كلمات المرور",
      "This policy sets the rules for account passwords.",
      "تحدد هذه السياسة قواعد كلمات مرور الحسابات.",
      [("Passwords must be at least {length} characters and mix upper, lower, number and symbol.",
        "يجب أن تكون كلمة المرور {length} حرفًا على الأقل وتخلط بين الأحرف الكبيرة والصغيرة والأرقام والرموز."),
       ("Passwords for systems with sensitive data must be changed every {rotate} days.",
        "يجب تغيير كلمات مرور الأنظمة التي تحتوي بيانات حساسة كل {rotate} يومًا."),
       ("Never share a password. IT staff will never ask for it.",
        "لا تشارك كلمة المرور مطلقًا. لن يطلبها موظفو تقنية المعلومات أبدًا."),
       ("A suspected compromise must be reported and the password changed immediately.",
        "يجب الإبلاغ عن أي اشتباه في اختراق وتغيير كلمة المرور فورًا.")],
      {"length": [12, 14, 16], "rotate": [60, 90, 180]},
      [("How long must a password be?", "ما الحد الأدنى لطول كلمة المرور؟",
        "At least {length} characters.", "{length} حرفًا على الأقل."),
       ("How often must sensitive passwords change?", "كم مرة تُغيَّر كلمات المرور الحساسة؟",
        "Every {rotate} days.", "كل {rotate} يومًا.")],
      ["mfa", "system_access", "security_incident"]),

    T("mfa", "IT",
      "Multi-Factor Authentication Policy", "سياسة المصادقة متعددة العوامل",
      "This policy requires a second factor for sign-in.",
      "تفرض هذه السياسة عاملًا ثانيًا لتسجيل الدخول.",
      [("MFA is required for email, the VPN and any system holding client or financial data.",
        "المصادقة متعددة العوامل مطلوبة للبريد وشبكة VPN وأي نظام يحتوي بيانات العملاء أو البيانات المالية."),
       ("The approved second factor is the authenticator app; SMS is allowed only as a fallback.",
        "العامل الثاني المعتمد هو تطبيق المصادقة، ويُسمح بالرسائل النصية كبديل فقط."),
       ("Lost devices must be reported to the service desk within {report_hours} hours.",
        "يجب الإبلاغ عن الأجهزة المفقودة إلى مكتب الخدمة خلال {report_hours} ساعة.")],
      {"report_hours": [1, 2, 4]},
      [("Is SMS allowed for MFA?", "هل يُسمح بالرسائل النصية للمصادقة؟",
        "Only as a fallback; the app is preferred.", "كبديل فقط، والتطبيق هو المفضل.")],
      ["password_policy", "vpn_remote_access"]),

    T("system_access", "IT",
      "System Access Request Policy", "سياسة طلب الوصول إلى الأنظمة",
      "This policy explains how access to systems is requested and reviewed.",
      "توضح هذه السياسة كيفية طلب الوصول إلى الأنظمة ومراجعته.",
      [("Access is requested through the service desk, stating the system and the business reason.",
        "يُطلب الوصول عبر مكتب الخدمة مع ذكر النظام والسبب العملي."),
       ("Access to client or financial data needs the manager and the system owner to approve.",
        "يتطلب الوصول إلى بيانات العملاء أو البيانات المالية موافقة المدير ومالك النظام."),
       ("Access unused for {revoke} days may be revoked and must be requested again.",
        "قد يُلغى الوصول غير المستخدم لمدة {revoke} يومًا ويجب طلبه من جديد."),
       ("All access is reviewed every quarter.",
        "يُراجع كل وصول كل ربع سنة.")],
      {"revoke": [60, 90]},
      [("How do I get access to a system?", "كيف أحصل على وصول إلى نظام؟",
        "Request it through the service desk with a business reason.",
        "اطلبه عبر مكتب الخدمة مع سبب عملي.")],
      ["password_policy", "acceptable_use", "onboarding"]),

    T("acceptable_use", "IT",
      "Acceptable Use Policy", "سياسة الاستخدام المقبول",
      "This policy sets what company devices and networks may be used for.",
      "تحدد هذه السياسة ما يمكن استخدام أجهزة الشركة وشبكاتها من أجله.",
      [("Company systems are for work. Limited personal use is allowed if it does not break policy or law.",
        "أنظمة الشركة للعمل. يُسمح باستخدام شخصي محدود إن لم يخالف السياسة أو القانون."),
       ("Installing unlicensed software or bypassing security controls is prohibited.",
        "يُحظر تثبيت برامج غير مرخصة أو تجاوز ضوابط الأمن."),
       ("The company may monitor use of its systems for security and compliance.",
        "يجوز للشركة مراقبة استخدام أنظمتها لأغراض الأمن والامتثال.")],
      {},
      [("Can I use my work laptop for personal things?", "هل يمكنني استخدام حاسوب العمل لأمور شخصية؟",
        "Limited personal use is allowed if it does not break policy or law.",
        "يُسمح باستخدام شخصي محدود إن لم يخالف السياسة أو القانون.")],
      ["software_licensing", "byod"], office_specific=False),

    T("byod", "IT",
      "Bring Your Own Device Policy", "سياسة استخدام الأجهزة الشخصية",
      "This policy covers using personal phones or laptops for work.",
      "تغطي هذه السياسة استخدام الهواتف أو الحواسيب الشخصية للعمل.",
      [("Personal devices may access email and calendar only, and must be enrolled in mobile management.",
        "يجوز للأجهزة الشخصية الوصول إلى البريد والتقويم فقط، ويجب تسجيلها في إدارة الأجهزة المحمولة."),
       ("The device must have a screen lock and up-to-date operating system.",
        "يجب أن يكون بالجهاز قفل شاشة ونظام تشغيل محدث."),
       ("The company can remotely wipe work data from a lost or stolen device.",
        "يمكن للشركة مسح بيانات العمل عن بُعد من جهاز مفقود أو مسروق.")],
      {},
      [("Can I read work email on my own phone?", "هل يمكنني قراءة بريد العمل على هاتفي؟",
        "Yes, if it is enrolled in mobile management and has a screen lock.",
        "نعم، إذا كان مسجلًا في إدارة الأجهزة وبه قفل شاشة.")],
      ["acceptable_use", "mfa"], office_specific=False),

    T("vpn_remote_access", "IT",
      "VPN and Remote Access Policy", "سياسة الوصول عن بُعد و VPN",
      "This policy explains how to connect to internal systems from outside the office.",
      "توضح هذه السياسة كيفية الاتصال بالأنظمة الداخلية من خارج المكتب.",
      [("Internal systems are reachable from outside only through the approved VPN with MFA.",
        "لا يمكن الوصول إلى الأنظمة الداخلية من الخارج إلا عبر VPN المعتمدة مع المصادقة متعددة العوامل."),
       ("Split tunnelling is disabled; all traffic goes through the VPN while connected.",
        "التوجيه المنقسم معطّل، وكل حركة البيانات تمر عبر VPN أثناء الاتصال."),
       ("Sessions time out after {timeout} minutes of inactivity.",
        "تنتهي الجلسات بعد {timeout} دقيقة من عدم النشاط.")],
      {"timeout": [15, 30, 60]},
      [("How do I reach internal systems from home?", "كيف أصل إلى الأنظمة الداخلية من المنزل؟",
        "Through the approved VPN with MFA.", "عبر VPN المعتمدة مع المصادقة متعددة العوامل.")],
      ["remote_work", "mfa"]),

    T("email_usage", "IT",
      "Email Usage Policy", "سياسة استخدام البريد الإلكتروني",
      "This policy covers correct use of company email.",
      "تغطي هذه السياسة الاستخدام الصحيح لبريد الشركة.",
      [("Company email is for business communication and is a company record.",
        "بريد الشركة مخصص للتواصل المهني ويُعد سجلًا للشركة."),
       ("Do not auto-forward company email to a personal account.",
        "لا تُعِد توجيه بريد الشركة تلقائيًا إلى حساب شخصي."),
       ("External emails with attachments over {size} MB should use the file-share link instead.",
        "ينبغي للرسائل الخارجية بمرفقات تزيد عن {size} ميجابايت استخدام رابط المشاركة بدلًا من ذلك.")],
      {"size": [10, 20, 25]},
      [("Can I forward my work email to Gmail?", "هل يمكنني توجيه بريد العمل إلى Gmail؟",
        "No. Auto-forwarding to a personal account is not allowed.",
        "لا. لا يُسمح بإعادة التوجيه التلقائي إلى حساب شخصي.")],
      ["acceptable_use", "phishing_reporting", "records_retention"], office_specific=False),

    T("data_backup", "IT",
      "Data Backup Policy", "سياسة النسخ الاحتياطي للبيانات",
      "This policy explains what is backed up and how often.",
      "توضح هذه السياسة ما يُنسخ احتياطيًا وكم مرة.",
      [("Files on approved company storage are backed up automatically every {freq}.",
        "تُنسخ الملفات على تخزين الشركة المعتمد احتياطيًا تلقائيًا كل {freq}."),
       ("Backups are kept for {retention} days and are tested every quarter.",
        "تُحفظ النسخ الاحتياطية لمدة {retention} يومًا وتُختبر كل ربع سنة."),
       ("Files kept only on a local laptop are not backed up and may be lost.",
        "الملفات المحفوظة على الحاسوب المحلي فقط لا تُنسخ احتياطيًا وقد تُفقد.")],
      {"freq": ["night", "4 hours"], "retention": [30, 60, 90]},
      [("Are files on my laptop backed up?", "هل تُنسخ ملفات حاسوبي احتياطيًا؟",
        "Only if they are on approved company storage.", "فقط إذا كانت على تخزين الشركة المعتمد.")],
      ["encryption", "records_retention"], office_specific=False),

    T("software_licensing", "IT",
      "Software Licensing Policy", "سياسة تراخيص البرمجيات",
      "This policy explains which software may be installed.",
      "توضح هذه السياسة البرمجيات التي يمكن تثبيتها.",
      [("Only software approved and licensed by IT may be installed on company devices.",
        "لا يجوز تثبيت إلا البرمجيات المعتمدة والمرخصة من تقنية المعلومات على أجهزة الشركة."),
       ("Free or trial software may be used for evaluation for up to {trial} days, then licensed or removed.",
        "يجوز استخدام البرمجيات المجانية أو التجريبية للتقييم حتى {trial} يومًا، ثم تُرخَّص أو تُزال."),
       ("Using pirated software is a serious policy violation.",
        "استخدام البرمجيات المقرصنة مخالفة جسيمة للسياسة.")],
      {"trial": [14, 30]},
      [("Can I use a free trial tool at work?", "هل يمكنني استخدام أداة تجريبية مجانية في العمل؟",
        "For up to {trial} days, then it must be licensed or removed.",
        "حتى {trial} يومًا، ثم يجب ترخيصها أو إزالتها.")],
      ["acceptable_use", "hardware_request"], office_specific=False),

    T("hardware_request", "IT",
      "Hardware Request Policy", "سياسة طلب الأجهزة",
      "This policy explains how to request laptops, monitors and phones.",
      "توضح هذه السياسة كيفية طلب الحواسيب والشاشات والهواتف.",
      [("Standard equipment is requested through the asset portal with a business justification.",
        "تُطلب المعدات القياسية عبر بوابة الأصول مع مبرر عملي."),
       ("Standard requests are approved by the manager and fulfilled within {fulfil} working days.",
        "تُعتمد الطلبات القياسية من المدير وتُنفَّذ خلال {fulfil} أيام عمل."),
       ("Non-standard items also need IT approval and may take longer.",
        "تتطلب العناصر غير القياسية موافقة تقنية المعلومات وقد تستغرق وقتًا أطول.")],
      {"fulfil": [3, 5, 10]},
      [("How long to get a second monitor?", "كم يستغرق الحصول على شاشة ثانية؟",
        "Standard requests are fulfilled within {fulfil} working days.",
        "تُنفَّذ الطلبات القياسية خلال {fulfil} أيام عمل.")],
      ["asset_return", "software_licensing", "workspace_ergonomics"]),

    T("asset_return", "IT",
      "Asset Return Policy", "سياسة إعادة الأصول",
      "This policy covers returning company equipment.",
      "تغطي هذه السياسة إعادة معدات الشركة.",
      [("All company equipment remains company property and is returned on the last working day.",
        "تبقى جميع معدات الشركة ملكًا لها وتُعاد في آخر يوم عمل."),
       ("Damaged or missing items beyond normal wear may be charged to the employee.",
        "قد تُحمَّل العناصر التالفة أو المفقودة بما يتجاوز الاستهلاك الطبيعي على الموظف."),
       ("Equipment for a role change is swapped through the service desk.",
        "تُبدَّل المعدات عند تغيير الدور عبر مكتب الخدمة.")],
      {},
      [("When do I return my laptop?", "متى أُعيد حاسوبي؟",
        "On your last working day.", "في آخر يوم عمل لك.")],
      ["offboarding", "hardware_request"], office_specific=False),

    T("security_incident", "IT",
      "Security Incident Reporting Policy", "سياسة الإبلاغ عن الحوادث الأمنية",
      "This policy explains how to report a suspected security problem.",
      "توضح هذه السياسة كيفية الإبلاغ عن مشكلة أمنية مشتبه بها.",
      [("Report any suspected breach, lost device or leaked data to the security team within {hours} hour(s).",
        "أبلغ فريق الأمن عن أي اختراق مشتبه به أو جهاز مفقود أو تسرب بيانات خلال {hours} ساعة."),
       ("Do not try to investigate or fix it yourself; preserve evidence.",
        "لا تحاول التحقيق أو الإصلاح بنفسك، واحفظ الأدلة."),
       ("A written summary is filed within {summary} working day(s) of the report.",
        "يُقدَّم ملخص مكتوب خلال {summary} يوم عمل من البلاغ.")],
      {"hours": [1, 2], "summary": [1, 2]},
      [("How fast must I report a lost work phone?", "بأي سرعة يجب الإبلاغ عن فقدان هاتف العمل؟",
        "Within {hours} hour(s).", "خلال {hours} ساعة.")],
      ["phishing_reporting", "data_privacy", "mfa"], office_specific=False),

    T("phishing_reporting", "IT",
      "Phishing Reporting Policy", "سياسة الإبلاغ عن التصيد",
      "This policy explains what to do with a suspicious email.",
      "توضح هذه السياسة ما يجب فعله برسالة مشبوهة.",
      [("Do not click links or open attachments in a suspicious email.",
        "لا تنقر الروابط ولا تفتح المرفقات في رسالة مشبوهة."),
       ("Use the \"Report Phishing\" button, or forward the email to phishing@{domain}.",
        "استخدم زر \"الإبلاغ عن تصيد\"، أو أعد توجيه الرسالة إلى phishing@{domain}."),
       ("If you already clicked, disconnect from the network and call the service desk.",
        "إذا نقرت بالفعل، افصل الجهاز عن الشبكة واتصل بمكتب الخدمة.")],
      {},
      [("What do I do with a phishing email?", "ماذا أفعل برسالة تصيد؟",
        "Report it with the button or forward it to phishing@{domain}. Do not click anything.",
        "أبلغ عنها بالزر أو أعد توجيهها إلى phishing@{domain}. لا تنقر أي شيء.")],
      ["security_incident", "email_usage"], office_specific=False),

    T("encryption", "IT",
      "Encryption Policy", "سياسة التشفير",
      "This policy states where encryption is required.",
      "تحدد هذه السياسة أين يُطلب التشفير.",
      [("All company laptops and phones must have full-disk encryption enabled.",
        "يجب تفعيل تشفير القرص الكامل على جميع حواسيب وهواتف الشركة."),
       ("Client or financial data sent outside the company must be encrypted or sent via the secure portal.",
        "يجب تشفير بيانات العملاء أو البيانات المالية المرسلة خارج الشركة أو إرسالها عبر البوابة الآمنة."),
       ("Encryption keys are managed by IT and never shared by email.",
        "تُدار مفاتيح التشفير من تقنية المعلومات ولا تُشارك عبر البريد أبدًا.")],
      {},
      [("Is my work laptop encrypted?", "هل حاسوب عملي مشفّر؟",
        "It must have full-disk encryption enabled.", "يجب أن يكون تشفير القرص الكامل مفعّلًا.")],
      ["data_backup", "data_privacy"], office_specific=False),

    T("clear_desk", "IT",
      "Clear Desk and Screen Policy", "سياسة المكتب والشاشة النظيفين",
      "This policy protects information left on desks and screens.",
      "تحمي هذه السياسة المعلومات المتروكة على المكاتب والشاشات.",
      [("Lock your screen whenever you leave your desk.",
        "اقفل شاشتك كلما غادرت مكتبك."),
       ("Documents with client or personal data are locked away at the end of the day.",
        "تُقفل المستندات التي تحتوي بيانات العملاء أو البيانات الشخصية في نهاية اليوم."),
       ("Printouts left on the printer for more than {minutes} minutes are shredded.",
        "تُتلف المطبوعات المتروكة على الطابعة أكثر من {minutes} دقيقة.")],
      {"minutes": [15, 30, 60]},
      [("Do I need to lock my screen for a short break?", "هل أقفل الشاشة لاستراحة قصيرة؟",
        "Yes, whenever you leave your desk.", "نعم، كلما غادرت مكتبك.")],
      ["printing", "data_privacy"]),

    T("printing", "IT",
      "Printing Policy", "سياسة الطباعة",
      "This policy covers secure and economical printing.",
      "تغطي هذه السياسة الطباعة الآمنة والاقتصادية.",
      [("Printing needs a badge tap at the printer to release the job.",
        "تتطلب الطباعة تمرير البطاقة عند الطابعة لإطلاق المهمة."),
       ("Default printing is double-sided and black-and-white.",
        "الطباعة الافتراضية على الوجهين وبالأبيض والأسود."),
       ("Unreleased jobs are deleted after {hours} hours.",
        "تُحذف المهام غير المُطلقة بعد {hours} ساعة.")],
      {"hours": [2, 4, 24]},
      [("Why didn't my document print straight away?", "لماذا لم تُطبع وثيقتي فورًا؟",
        "You must tap your badge at the printer to release it.",
        "يجب تمرير بطاقتك عند الطابعة لإطلاقها.")],
      ["clear_desk"]),

    T("help_desk_sla", "IT",
      "IT Support Service Levels", "مستويات خدمة الدعم التقني",
      "This document sets response times for IT support tickets.",
      "تحدد هذه الوثيقة أوقات الاستجابة لتذاكر الدعم التقني.",
      [("Standard tickets are responded to within {standard} business day(s).",
        "يُرد على التذاكر العادية خلال {standard} يوم عمل."),
       ("Critical issues, such as a full outage, are responded to within {critical} hour(s) during business hours.",
        "يُرد على الأعطال الحرجة، مثل الانقطاع الكامل، خلال {critical} ساعة خلال ساعات العمل."),
       ("Raise a ticket through the service desk portal, not by contacting staff directly.",
        "افتح تذكرة عبر بوابة مكتب الخدمة وليس بالتواصل مع الموظفين مباشرة.")],
      {"standard": [1, 2], "critical": [1, 2, 4]},
      [("How fast is a critical IT issue handled?", "بأي سرعة يُعالج عطل تقني حرج؟",
        "Within {critical} hour(s) during business hours.", "خلال {critical} ساعة خلال ساعات العمل.")],
      ["system_access", "hardware_request"]),

    # ---------------- FIN ----------------
    T("expense_reimbursement", "FIN",
      "Expense Reimbursement Policy", "سياسة استرداد المصروفات",
      "This policy explains how to claim back approved business expenses.",
      "توضح هذه السياسة كيفية استرداد المصروفات العملية المعتمدة.",
      [("Claims are submitted through the finance portal within {days} days of the expense, with a receipt.",
        "تُقدَّم المطالبات عبر بوابة المالية خلال {days} يومًا من الصرف مع إيصال."),
       ("Any single item over {threshold} {currency} needs the manager's approval before it is spent.",
        "يتطلب أي بند مفرد يتجاوز {threshold} {currency} موافقة المدير قبل الصرف."),
       ("Alcohol, personal entertainment and traffic fines are never reimbursed.",
        "لا تُسترد أبدًا تكاليف الكحول والترفيه الشخصي والمخالفات المرورية."),
       ("Approved claims are paid with the next payroll run.",
        "تُدفع المطالبات المعتمدة مع دورة الرواتب التالية.")],
      {"days": [14, 30, 45], "threshold": [200, 500, 1000], "currency": ["SAR", "AED", "EGP"]},
      [("How long do I have to claim an expense?", "كم لديّ من الوقت للمطالبة بمصروف؟",
        "Within {days} days of the expense.", "خلال {days} يومًا من الصرف."),
       ("Do I need approval before spending {threshold} {currency}?",
        "هل أحتاج موافقة قبل صرف {threshold} {currency}؟",
        "Yes, single items over {threshold} {currency} need manager approval first.",
        "نعم، البنود المفردة التي تتجاوز {threshold} {currency} تتطلب موافقة المدير أولًا.")],
      ["corporate_card", "travel_booking", "per_diem"]),

    T("corporate_card", "FIN",
      "Corporate Card Policy", "سياسة بطاقة الشركة",
      "This policy covers the company credit card.",
      "تغطي هذه السياسة بطاقة ائتمان الشركة.",
      [("The card is for business expenses only; personal use is not allowed.",
        "البطاقة للمصروفات العملية فقط، ولا يُسمح بالاستخدام الشخصي."),
       ("Receipts and a short note are uploaded within {days} days of each transaction.",
        "تُرفع الإيصالات مع ملاحظة قصيرة خلال {days} أيام من كل معاملة."),
       ("Undocumented transactions after {grace} days may be deducted from salary.",
        "قد تُخصم المعاملات غير الموثقة بعد {grace} يومًا من الراتب.")],
      {"days": [5, 7], "grace": [30, 45]},
      [("Can I use the corporate card for a personal purchase?", "هل يمكنني استخدام بطاقة الشركة لشراء شخصي؟",
        "No. It is for business expenses only.", "لا. هي للمصروفات العملية فقط.")],
      ["expense_reimbursement", "travel_booking"], office_specific=False),

    T("travel_booking", "FIN",
      "Business Travel Booking Policy", "سياسة حجز السفر للعمل",
      "This policy explains how to book flights and hotels for work trips.",
      "توضح هذه السياسة كيفية حجز الرحلات والفنادق لرحلات العمل.",
      [("All travel is booked through the corporate travel portal, not directly with airlines or hotels.",
        "يُحجز كل سفر عبر بوابة السفر المؤسسية وليس مباشرة مع شركات الطيران أو الفنادق."),
       ("Economy class is standard; flights over {long_hours} hours may be premium economy with manager approval.",
        "الدرجة السياحية هي المعيار، والرحلات التي تتجاوز {long_hours} ساعة يمكن أن تكون سياحية مميزة بموافقة المدير."),
       ("Hotels must not exceed the nightly cap for the destination shown in the portal.",
        "يجب ألا تتجاوز الفنادق الحد الليلي للوجهة المبيّن في البوابة."),
       ("Book at least {advance} days ahead where possible.",
        "احجز قبل {advance} أيام على الأقل حيثما أمكن.")],
      {"long_hours": [5, 6, 8], "advance": [7, 14]},
      [("Can I book my own flight directly with the airline?", "هل يمكنني حجز رحلتي مباشرة مع شركة الطيران؟",
        "No. All travel is booked through the corporate travel portal.",
        "لا. يُحجز كل سفر عبر بوابة السفر المؤسسية.")],
      ["expense_reimbursement", "per_diem", "corporate_card"], office_specific=False),

    T("per_diem", "FIN",
      "Per Diem Policy", "سياسة بدل الإقامة اليومي",
      "This policy sets the daily allowance for meals and incidentals while travelling.",
      "تحدد هذه السياسة البدل اليومي للوجبات والنثريات أثناء السفر.",
      [("The per diem covers meals and small incidentals and removes the need for individual meal receipts.",
        "يغطي البدل اليومي الوجبات والنثريات الصغيرة ويغني عن إيصالات الوجبات المنفردة."),
       ("The daily rate is {rate} {currency} for domestic travel and {intl_rate} {currency} for international travel.",
        "المعدل اليومي {rate} {currency} للسفر الداخلي و{intl_rate} {currency} للسفر الدولي."),
       ("The per diem is not paid for days where all meals are provided.",
        "لا يُدفع البدل اليومي عن الأيام التي تُقدَّم فيها جميع الوجبات.")],
      {"rate": [150, 200, 250], "intl_rate": [300, 400, 500], "currency": ["SAR", "AED", "EGP"]},
      [("Do I keep meal receipts if I get a per diem?", "هل أحتفظ بإيصالات الوجبات مع البدل اليومي؟",
        "No. The per diem removes the need for individual meal receipts.",
        "لا. يغني البدل اليومي عن إيصالات الوجبات المنفردة.")],
      ["travel_booking", "expense_reimbursement"], office_specific=False),

    T("procurement", "FIN",
      "Procurement Policy", "سياسة الشراء",
      "This policy explains how the company buys goods and services.",
      "توضح هذه السياسة كيفية شراء الشركة للسلع والخدمات.",
      [("A purchase request is raised before any commitment to a supplier.",
        "يُرفع طلب شراء قبل أي التزام تجاه مورد."),
       ("Purchases up to {t1} {currency} need one quote; above {t1} need {quotes} quotes.",
        "المشتريات حتى {t1} {currency} تحتاج عرض سعر واحد، وما فوق ذلك يحتاج {quotes} عروض."),
       ("Purchases over {t2} {currency} go to the Procurement Committee.",
        "المشتريات التي تتجاوز {t2} {currency} تُحال إلى لجنة المشتريات.")],
      {"t1": [5000, 10000], "quotes": [2, 3], "t2": [50000, 100000], "currency": ["SAR", "AED", "EGP"]},
      [("How many quotes do I need for a {t1} {currency} purchase?",
        "كم عرض سعر أحتاج لشراء بقيمة {t1} {currency}؟",
        "One quote up to {t1} {currency}.", "عرض واحد حتى {t1} {currency}.")],
      ["purchase_request", "supplier_selection", "budget_approval"], office_specific=False),

    T("invoice_processing", "FIN",
      "Invoice Processing Policy", "سياسة معالجة الفواتير",
      "This policy explains how supplier invoices are approved and paid.",
      "توضح هذه السياسة كيفية اعتماد فواتير الموردين ودفعها.",
      [("An invoice is paid only against an approved purchase order and a goods-received note.",
        "تُدفع الفاتورة فقط مقابل أمر شراء معتمد وإشعار استلام بضاعة."),
       ("Standard payment terms are {terms} days from the invoice date.",
        "شروط الدفع القياسية {terms} يومًا من تاريخ الفاتورة."),
       ("Invoices without a purchase order are returned to the supplier.",
        "تُعاد الفواتير التي بلا أمر شراء إلى المورد.")],
      {"terms": [30, 45, 60]},
      [("What are the standard payment terms?", "ما شروط الدفع القياسية؟",
        "{terms} days from the invoice date.", "{terms} يومًا من تاريخ الفاتورة.")],
      ["procurement", "vendor_onboarding"], office_specific=False),

    T("budget_approval", "FIN",
      "Budget Approval Policy", "سياسة اعتماد الميزانية",
      "This policy sets who can approve spending against a budget.",
      "تحدد هذه السياسة من يمكنه اعتماد الإنفاق مقابل الميزانية.",
      [("Managers may approve spend within their budget up to {mgr} {currency} per item.",
        "يجوز للمديرين اعتماد الإنفاق ضمن ميزانيتهم حتى {mgr} {currency} لكل بند."),
       ("Department heads approve up to {head} {currency}; anything higher needs the CFO.",
        "يعتمد رؤساء الأقسام حتى {head} {currency}، وما يزيد يتطلب المدير المالي."),
       ("Spend not in the approved budget needs a business case.",
        "الإنفاق غير المدرج في الميزانية المعتمدة يتطلب مبرر عمل.")],
      {"mgr": [5000, 10000], "head": [50000, 100000], "currency": ["SAR", "AED", "EGP"]},
      [("What can my manager approve?", "ما الذي يمكن لمديري اعتماده؟",
        "Up to {mgr} {currency} per item within budget.", "حتى {mgr} {currency} لكل بند ضمن الميزانية.")],
      ["procurement", "purchase_thresholds"], office_specific=False),

    T("petty_cash", "FIN",
      "Petty Cash Policy", "سياسة المصروفات النثرية",
      "This policy covers small cash payments held at each office.",
      "تغطي هذه السياسة المدفوعات النقدية الصغيرة المحتفظ بها في كل مكتب.",
      [("Petty cash is for small items only, up to {limit} {currency} per payment.",
        "المصروفات النثرية للبنود الصغيرة فقط، حتى {limit} {currency} لكل دفعة."),
       ("Every payment needs a receipt and a signed voucher.",
        "تحتاج كل دفعة إلى إيصال وسند موقّع."),
       ("The float is reconciled and topped up every {cycle}.",
        "تُسوَّى العهدة وتُجدَّد كل {cycle}.")],
      {"limit": [100, 200, 300], "currency": ["SAR", "AED", "EGP"], "cycle": ["week", "month"]},
      [("What is the petty cash limit per payment?", "ما حد الدفعة الواحدة للمصروفات النثرية؟",
        "Up to {limit} {currency}.", "حتى {limit} {currency}.")],
      ["expense_reimbursement"]),

    T("payroll_schedule", "FIN",
      "Payroll Schedule Policy", "سياسة جدول الرواتب",
      "This document states when salaries are paid.",
      "تحدد هذه الوثيقة موعد صرف الرواتب.",
      [("Salaries are paid on the {payday} of each month, or the previous working day if that is a weekend or holiday.",
        "تُصرف الرواتب في {payday} من كل شهر، أو يوم العمل السابق إذا صادف عطلة نهاية الأسبوع أو عطلة رسمية."),
       ("Changes to bank details must reach Payroll {cutoff} working days before payday.",
        "يجب أن تصل تغييرات بيانات البنك إلى الرواتب قبل {cutoff} أيام عمل من موعد الصرف."),
       ("Payslips are available in the HR system on payday.",
        "تتوفر قسائم الراتب في نظام الموارد البشرية يوم الصرف.")],
      {"payday": ["25th", "27th", "last day"], "cutoff": [3, 5]},
      [("When are salaries paid?", "متى تُصرف الرواتب؟",
        "On the {payday} of each month.", "في {payday} من كل شهر.")],
      ["salary_review", "tax_documents"]),

    T("tax_documents", "FIN",
      "Tax Documents Policy", "سياسة المستندات الضريبية",
      "This policy explains how to get annual tax and salary certificates.",
      "توضح هذه السياسة كيفية الحصول على الشهادات الضريبية وشهادات الراتب السنوية.",
      [("Salary certificates are requested through the HR system and issued within {days} working days.",
        "تُطلب شهادات الراتب عبر نظام الموارد البشرية وتُصدر خلال {days} أيام عمل."),
       ("Annual tax summaries are published in the HR system by {month} each year.",
        "تُنشر الملخصات الضريبية السنوية في نظام الموارد البشرية بحلول {month} من كل عام.")],
      {"days": [2, 3, 5], "month": ["January", "February", "March"]},
      [("How do I get a salary certificate?", "كيف أحصل على شهادة راتب؟",
        "Request it in the HR system; it is issued within {days} working days.",
        "اطلبها عبر نظام الموارد البشرية، وتُصدر خلال {days} أيام عمل.")],
      ["payroll_schedule"], office_specific=False),

    T("vendor_onboarding", "FIN",
      "Vendor Onboarding Policy", "سياسة اعتماد الموردين",
      "This policy explains how a new supplier is set up for payment.",
      "توضح هذه السياسة كيفية إعداد مورد جديد للدفع.",
      [("A new vendor provides a trade licence, bank letter and tax registration before setup.",
        "يقدم المورد الجديد رخصة تجارية وخطاب بنك وتسجيلًا ضريبيًا قبل الإعداد."),
       ("Bank details are verified by a call-back to a number on the trade licence, not the invoice.",
        "تُتحقَّق بيانات البنك عبر اتصال مرتد برقم من الرخصة التجارية وليس من الفاتورة."),
       ("Vendor records are reviewed every {review_months} months.",
        "تُراجع سجلات الموردين كل {review_months} أشهر.")],
      {"review_months": [12, 24]},
      [("How are new vendor bank details checked?", "كيف تُتحقق بيانات بنك المورد الجديد؟",
        "By a call-back to a number on the trade licence, not the invoice.",
        "عبر اتصال مرتد برقم من الرخصة التجارية وليس من الفاتورة.")],
      ["invoice_processing", "anti_bribery"], office_specific=False),

    # ---------------- FAC ----------------
    T("meeting_room_booking", "FAC",
      "Meeting Room Booking Policy", "سياسة حجز قاعات الاجتماعات",
      "This policy explains how to reserve and use meeting rooms.",
      "توضح هذه السياسة كيفية حجز قاعات الاجتماعات واستخدامها.",
      [("Rooms are booked through the shared calendar. Walk-in use is allowed only if the room is free for the next {free} minutes.",
        "تُحجز القاعات عبر التقويم المشترك. يُسمح بالاستخدام الفوري فقط إذا كانت القاعة شاغرة لـ {free} دقيقة قادمة."),
       ("Bookings longer than {long_hours} hours need an office administrator to approve.",
        "تتطلب الحجوزات التي تتجاوز {long_hours} ساعة موافقة مسؤول المكتب."),
       ("If a meeting ends early or is cancelled, release the room in the calendar.",
        "إذا انتهى الاجتماع مبكرًا أو أُلغي، حرر القاعة في التقويم.")],
      {"free": [15, 30], "long_hours": [2, 3]},
      [("Can I use an empty meeting room without booking?", "هل يمكنني استخدام قاعة فارغة دون حجز؟",
        "Only if it is free for the next {free} minutes.", "فقط إذا كانت شاغرة لـ {free} دقيقة قادمة.")],
      ["visitor_management", "parking"]),

    T("parking", "FAC",
      "Parking Policy", "سياسة مواقف السيارات",
      "This policy explains how office parking is allocated.",
      "توضح هذه السياسة كيفية تخصيص مواقف المكتب.",
      [("Parking is limited. Spaces are assigned by the waiting list, reviewed every {review} months.",
        "المواقف محدودة. تُخصَّص الأماكن حسب قائمة الانتظار، وتُراجع كل {review} أشهر."),
       ("Reserved spaces are for visitors and accessibility needs only.",
        "الأماكن المحجوزة للزوار وذوي الاحتياجات الخاصة فقط."),
       ("Parking in a reserved space without a permit may lead to the car being clamped.",
        "قد يؤدي الوقوف في مكان محجوز دون تصريح إلى تكبيل السيارة.")],
      {"review": [3, 6, 12]},
      [("How do I get a parking space?", "كيف أحصل على موقف؟",
        "Join the waiting list; spaces are reviewed every {review} months.",
        "انضم إلى قائمة الانتظار، وتُراجع الأماكن كل {review} أشهر.")],
      ["building_access"]),

    T("building_access", "FAC",
      "Building Access Policy", "سياسة الدخول إلى المبنى",
      "This policy covers access cards and entry to the office.",
      "تغطي هذه السياسة بطاقات الدخول والدخول إلى المكتب.",
      [("Every employee has an access card. Do not let anyone \"tailgate\" in behind you.",
        "لكل موظف بطاقة دخول. لا تسمح لأحد بالدخول خلفك دون تمرير بطاقته."),
       ("Lost cards are reported the same day; a replacement costs {fee} {currency}.",
        "تُبلَّغ البطاقات المفقودة في اليوم نفسه، وتكلفة البدل {fee} {currency}."),
       ("After-hours access is logged and may need manager approval.",
        "يُسجَّل الدخول بعد ساعات العمل وقد يتطلب موافقة المدير.")],
      {"fee": [25, 50, 100], "currency": ["SAR", "AED", "EGP"]},
      [("What if I lose my access card?", "ماذا لو فقدت بطاقة الدخول؟",
        "Report it the same day; a replacement costs {fee} {currency}.",
        "أبلغ عنها في اليوم نفسه، وتكلفة البدل {fee} {currency}.")],
      ["parking", "visitor_management"]),

    T("visitor_management", "FAC",
      "Visitor Management Policy", "سياسة إدارة الزوار",
      "This policy covers hosting visitors in the office.",
      "تغطي هذه السياسة استضافة الزوار في المكتب.",
      [("Visitors are registered in advance and are met and escorted by their host.",
        "يُسجَّل الزوار مسبقًا ويُستقبلون ويُرافقون من مضيفهم."),
       ("Visitors sign in at reception, wear a badge, and sign out when leaving.",
        "يوقّع الزوار عند الاستقبال ويرتدون بطاقة ويوقّعون عند المغادرة."),
       ("Visitors are not left alone in areas with client or personal data.",
        "لا يُترك الزوار وحدهم في مناطق تحتوي بيانات العملاء أو البيانات الشخصية.")],
      {},
      [("Do I need to register a visitor in advance?", "هل يجب تسجيل الزائر مسبقًا؟",
        "Yes, and you must meet and escort them.", "نعم، ويجب أن تستقبله وترافقه.")],
      ["building_access", "data_privacy"]),

    T("health_safety", "FAC",
      "Workplace Health and Safety Policy", "سياسة الصحة والسلامة في مكان العمل",
      "This policy sets basic health and safety rules for the office.",
      "تحدد هذه السياسة قواعد الصحة والسلامة الأساسية للمكتب.",
      [("Any workplace injury, however minor, is reported to the office manager within {hours} hours using the incident form.",
        "يُبلَّغ عن أي إصابة عمل مهما كانت بسيطة إلى مدير المكتب خلال {hours} ساعة باستخدام نموذج الحادث."),
       ("Keep walkways and fire exits clear at all times.",
        "أبقِ الممرات ومخارج الطوارئ خالية في جميع الأوقات."),
       ("Employees with symptoms of a contagious illness work from home instead of coming in.",
        "يعمل الموظفون الذين تظهر عليهم أعراض مرض مُعدٍ من المنزل بدلًا من الحضور.")],
      {"hours": [24, 48]},
      [("I cut my hand slightly at the office - do I report it?", "جرحت يدي جرحًا بسيطًا في المكتب - هل أُبلغ؟",
        "Yes, within {hours} hours using the incident form.",
        "نعم، خلال {hours} ساعة باستخدام نموذج الحادث.")],
      ["fire_safety", "first_aid", "workspace_ergonomics"]),

    T("fire_safety", "FAC",
      "Fire Safety Policy", "سياسة السلامة من الحرائق",
      "This policy covers fire drills and evacuation.",
      "تغطي هذه السياسة تدريبات الحريق والإخلاء.",
      [("Fire drills are held {drills} times a year and attendance is mandatory for everyone on site.",
        "تُجرى تدريبات الحريق {drills} مرات في السنة وحضورها إلزامي لكل من في الموقع."),
       ("On the alarm, leave by the nearest exit and go to the assembly point. Do not use lifts.",
        "عند الإنذار، غادر من أقرب مخرج واتجه إلى نقطة التجمع. لا تستخدم المصاعد."),
       ("Fire wardens check their area and report headcount at the assembly point.",
        "يتفقد مسؤولو الحريق مناطقهم ويبلّغون عدد الحاضرين عند نقطة التجمع.")],
      {"drills": [2, 3, 4]},
      [("How often are fire drills?", "كم مرة تُجرى تدريبات الحريق؟",
        "{drills} times a year.", "{drills} مرات في السنة.")],
      ["health_safety", "building_access"]),

    T("first_aid", "FAC",
      "First Aid Policy", "سياسة الإسعافات الأولية",
      "This policy covers first-aid cover in the office.",
      "تغطي هذه السياسة تغطية الإسعافات الأولية في المكتب.",
      [("Each floor has at least one trained first aider and a stocked first-aid kit.",
        "لكل طابق مسعف مدرب واحد على الأقل وحقيبة إسعافات مجهّزة."),
       ("First aiders are listed on the intranet and on the notice board by the lifts.",
        "يُدرج المسعفون في الشبكة الداخلية وعلى لوحة الإعلانات بجوار المصاعد."),
       ("Call emergency services first for anything serious, then a first aider.",
        "اتصل بخدمات الطوارئ أولًا لأي حالة خطيرة، ثم بمسعف.")],
      {},
      [("Where do I find a first aider?", "أين أجد مسعفًا؟",
        "On the intranet and the notice board by the lifts.",
        "في الشبكة الداخلية وعلى لوحة الإعلانات بجوار المصاعد.")],
      ["health_safety"], office_specific=False),

    T("workspace_ergonomics", "FAC",
      "Workspace Ergonomics Policy", "سياسة بيئة العمل المريحة",
      "This policy helps employees set up a healthy workspace.",
      "تساعد هذه السياسة الموظفين على إعداد مساحة عمل صحية.",
      [("A workstation assessment is offered to every new employee and on request.",
        "يُعرض تقييم محطة العمل على كل موظف جديد وعند الطلب."),
       ("The company provides an adjustable chair and, on assessment, extras such as a footrest or riser.",
        "توفر الشركة كرسيًا قابلًا للتعديل، وبناءً على التقييم إضافات مثل مسند قدم أو رافعة شاشة."),
       ("Remote employees can claim up to {claim} {currency} toward a home-office chair or desk.",
        "يمكن للموظفين عن بُعد المطالبة بما يصل إلى {claim} {currency} لكرسي أو مكتب منزلي.")],
      {"claim": [500, 1000, 1500], "currency": ["SAR", "AED", "EGP"]},
      [("Can I get help setting up my desk?", "هل يمكنني الحصول على مساعدة في إعداد مكتبي؟",
        "Yes, a workstation assessment is offered on request.",
        "نعم، يُعرض تقييم محطة العمل عند الطلب.")],
      ["hardware_request", "health_safety", "remote_work"]),

    T("kitchen_use", "FAC",
      "Kitchen and Break Area Policy", "سياسة المطبخ ومناطق الاستراحة",
      "This policy covers shared kitchens and break areas.",
      "تغطي هذه السياسة المطابخ ومناطق الاستراحة المشتركة.",
      [("Clean up after yourself; label personal food with your name and the date.",
        "نظّف بعد نفسك، وضع اسمك والتاريخ على طعامك الشخصي."),
       ("The fridge is emptied every {clear_day}; anything unlabelled is thrown away.",
        "يُفرَّغ الثلاجة كل {clear_day}، ويُرمى كل ما لا يحمل بطاقة."),
       ("Report broken appliances to facilities.",
        "أبلغ إدارة المرافق عن الأجهزة المعطلة.")],
      {"clear_day": ["Friday", "Thursday"]},
      [("When is the office fridge cleared?", "متى تُفرَّغ ثلاجة المكتب؟",
        "Every {clear_day}.", "كل {clear_day}.")],
      []),

    T("mail_courier", "FAC",
      "Mail and Courier Policy", "سياسة البريد والشحن",
      "This policy covers incoming and outgoing post and parcels.",
      "تغطي هذه السياسة البريد والطرود الواردة والصادرة.",
      [("Personal parcels should not be sent to the office address.",
        "لا ينبغي إرسال الطرود الشخصية إلى عنوان المكتب."),
       ("Outgoing business courier is booked through reception with a cost code.",
        "يُحجز الشحن التجاري الصادر عبر الاستقبال مع رمز تكلفة."),
       ("Reception holds parcels for {days} days before returning them to sender.",
        "يحتفظ الاستقبال بالطرود لمدة {days} أيام قبل إعادتها إلى المرسل.")],
      {"days": [5, 7, 10]},
      [("Can I ship a personal package to the office?", "هل يمكنني شحن طرد شخصي إلى المكتب؟",
        "No, personal parcels should not be sent to the office address.",
        "لا، لا ينبغي إرسال الطرود الشخصية إلى عنوان المكتب.")],
      []),

    # ---------------- LEG ----------------
    T("data_privacy", "LEG",
      "Data Privacy Policy", "سياسة خصوصية البيانات",
      "This policy covers how personal and client data must be handled.",
      "تغطي هذه السياسة كيفية التعامل مع البيانات الشخصية وبيانات العملاء.",
      [("Personal and client data is used only for the purpose it was collected for.",
        "تُستخدم البيانات الشخصية وبيانات العملاء فقط للغرض الذي جُمعت من أجله."),
       ("Do not store client or personal data on personal devices or personal cloud accounts.",
        "لا تخزّن بيانات العملاء أو البيانات الشخصية على أجهزة شخصية أو حسابات سحابية شخصية."),
       ("A suspected data breach is reported to the security team within {hours} hour(s).",
        "يُبلَّغ عن أي اشتباه في خرق للبيانات إلى فريق الأمن خلال {hours} ساعة."),
       ("Access to personal data is limited to those who need it for their work.",
        "يقتصر الوصول إلى البيانات الشخصية على من يحتاجها لعمله.")],
      {"hours": [1, 2, 4]},
      [("Where can I store customer data?", "أين يمكنني تخزين بيانات العملاء؟",
        "Only on approved company systems, never on personal devices.",
        "فقط على أنظمة الشركة المعتمدة، وليس على أجهزة شخصية."),
       ("How fast must a data breach be reported?", "بأي سرعة يجب الإبلاغ عن خرق البيانات؟",
        "Within {hours} hour(s) of discovering it.", "خلال {hours} ساعة من اكتشافه.")],
      ["confidentiality", "records_retention", "security_incident"], office_specific=False),

    T("confidentiality", "LEG",
      "Confidentiality Policy", "سياسة السرية",
      "This policy covers keeping company and client information confidential.",
      "تغطي هذه السياسة الحفاظ على سرية معلومات الشركة والعملاء.",
      [("Do not share confidential information outside the company without written approval.",
        "لا تشارك المعلومات السرية خارج الشركة دون موافقة كتابية."),
       ("Confidentiality continues after employment ends.",
        "تستمر السرية بعد انتهاء العمل."),
       ("Mark documents with the correct sensitivity label before sharing internally.",
        "ضع على المستندات تصنيف الحساسية الصحيح قبل مشاركتها داخليًا.")],
      {},
      [("Does confidentiality end when I leave?", "هل تنتهي السرية عند مغادرتي؟",
        "No. It continues after employment ends.", "لا. تستمر بعد انتهاء العمل.")],
      ["data_privacy", "intellectual_property", "external_communications"], office_specific=False),

    T("conflict_of_interest", "LEG",
      "Conflict of Interest Policy", "سياسة تضارب المصالح",
      "This policy covers situations where personal interests could affect work decisions.",
      "تغطي هذه السياسة الحالات التي قد تؤثر فيها المصالح الشخصية على قرارات العمل.",
      [("Disclose in writing any outside job, board seat, or financial interest in a supplier or client.",
        "أفصح كتابيًا عن أي عمل خارجي أو عضوية مجلس أو مصلحة مالية في مورد أو عميل."),
       ("Working for a direct competitor is not allowed while employed.",
        "لا يُسمح بالعمل لدى منافس مباشر أثناء التوظيف."),
       ("If a conflict arises in a decision, step back and let someone else handle it.",
        "إذا نشأ تضارب في قرار، تنحَّ ودع شخصًا آخر يتولاه.")],
      {},
      [("Do I have to tell anyone if I also work for another company?",
        "هل يجب أن أخبر أحدًا إذا كنت أعمل لدى شركة أخرى؟",
        "Yes, disclose it in writing. Working for a direct competitor is not allowed.",
        "نعم، أفصح عنه كتابيًا. العمل لدى منافس مباشر غير مسموح.")],
      ["code_of_conduct", "gifts_hospitality", "anti_bribery"], office_specific=False),

    T("anti_bribery", "LEG",
      "Anti-Bribery Policy", "سياسة مكافحة الرشوة",
      "This policy prohibits bribery and improper payments.",
      "تحظر هذه السياسة الرشوة والمدفوعات غير السليمة.",
      [("Never offer, give, request or accept a bribe, in any form, to win business or influence a decision.",
        "لا تعرض أو تعطِ أو تطلب أو تقبل رشوة بأي شكل للفوز بعمل أو التأثير في قرار."),
       ("Facilitation payments are not allowed, even where they are common.",
        "مدفوعات التيسير غير مسموحة حتى حيث تكون شائعة."),
       ("Any request for a bribe is reported to the Legal & Compliance Office immediately.",
        "يُبلَّغ فورًا عن أي طلب رشوة إلى مكتب الشؤون القانونية والامتثال.")],
      {},
      [("Are small facilitation payments allowed?", "هل تُسمح مدفوعات التيسير الصغيرة؟",
        "No, not even where they are common.", "لا، حتى حيث تكون شائعة.")],
      ["conflict_of_interest", "gifts_hospitality", "whistleblowing"], office_specific=False),

    T("whistleblowing", "LEG",
      "Whistleblowing Policy", "سياسة الإبلاغ عن المخالفات",
      "This policy explains how to report serious wrongdoing safely.",
      "توضح هذه السياسة كيفية الإبلاغ عن مخالفات جسيمة بأمان.",
      [("Serious concerns - fraud, safety risks, law breaking - can be raised through the confidential channel.",
        "يمكن رفع المخاوف الجسيمة - الاحتيال، مخاطر السلامة، مخالفة القانون - عبر القناة السرية."),
       ("Reports can be anonymous, and are handled by people independent of the area concerned.",
        "يمكن أن تكون البلاغات مجهولة، ويتولاها أشخاص مستقلون عن المجال المعني."),
       ("Retaliation against a good-faith reporter is a serious disciplinary matter.",
        "الانتقام من مُبلِّغ بحسن نية مسألة تأديبية جسيمة.")],
      {},
      [("Can I report fraud anonymously?", "هل يمكنني الإبلاغ عن احتيال بشكل مجهول؟",
        "Yes, through the confidential channel.", "نعم، عبر القناة السرية.")],
      ["anti_harassment", "anti_bribery", "code_of_conduct"], office_specific=False),

    T("records_retention", "LEG",
      "Records Retention Policy", "سياسة الاحتفاظ بالسجلات",
      "This policy states how long different records are kept.",
      "تحدد هذه السياسة مدة الاحتفاظ بأنواع السجلات المختلفة.",
      [("Financial records are kept for {fin_years} years; HR records for {hr_years} years after an employee leaves.",
        "تُحفظ السجلات المالية {fin_years} سنوات، وسجلات الموارد البشرية {hr_years} سنوات بعد مغادرة الموظف."),
       ("Records past their retention period are deleted in a controlled way, with a log.",
        "تُحذف السجلات بعد انتهاء مدة الاحتفاظ بطريقة منضبطة مع سجل."),
       ("Do not delete records that are subject to a legal hold.",
        "لا تحذف السجلات الخاضعة لتعليق قانوني.")],
      {"fin_years": [5, 7, 10], "hr_years": [2, 5, 7]},
      [("How long are financial records kept?", "كم مدة الاحتفاظ بالسجلات المالية؟",
        "{fin_years} years.", "{fin_years} سنوات.")],
      ["data_privacy", "data_backup", "email_usage"], office_specific=False),

    T("contract_signing_authority", "LEG",
      "Contract Signing Authority Policy", "سياسة صلاحية توقيع العقود",
      "This policy states who is allowed to sign contracts on behalf of the company.",
      "تحدد هذه السياسة من يُسمح له بتوقيع العقود باسم الشركة.",
      [("Only named signatories in the authority matrix may sign contracts for {org}.",
        "لا يجوز توقيع العقود باسم {org} إلا للمفوضين المسمّين في مصفوفة الصلاحيات."),
       ("Contracts over {value} {currency} or longer than {years} years need board approval.",
        "تتطلب العقود التي تتجاوز {value} {currency} أو تزيد مدتها عن {years} سنوات موافقة مجلس الإدارة."),
       ("Legal reviews every contract before signing.",
        "تراجع الإدارة القانونية كل عقد قبل التوقيع.")],
      {"value": [500000, 1000000], "years": [3, 5], "currency": ["SAR", "AED", "EGP"]},
      [("Can I sign a supplier contract myself?", "هل يمكنني توقيع عقد مورد بنفسي؟",
        "Only if you are a named signatory in the authority matrix.",
        "فقط إذا كنت مفوضًا مسمّى في مصفوفة الصلاحيات.")],
      ["procurement", "budget_approval"], office_specific=False),

    T("gifts_hospitality", "LEG",
      "Gifts and Hospitality Policy", "سياسة الهدايا والضيافة",
      "This policy sets limits on giving and receiving gifts and hospitality.",
      "تحدد هذه السياسة حدود تقديم وتلقي الهدايا والضيافة.",
      [("Gifts or hospitality worth more than {limit} {currency} must be declared in the register.",
        "يجب تسجيل الهدايا أو الضيافة التي تزيد قيمتها عن {limit} {currency} في السجل."),
       ("Never accept cash or a cash equivalent.",
        "لا تقبل نقودًا أو ما يعادلها أبدًا."),
       ("Do not give or accept anything around the time of a tender or contract decision.",
        "لا تعطِ أو تقبل أي شيء قرب موعد قرار مناقصة أو عقد.")],
      {"limit": [200, 300, 500], "currency": ["SAR", "AED", "EGP"]},
      [("Do I need to declare a client gift?", "هل يجب تسجيل هدية من عميل؟",
        "Yes, if it is worth more than {limit} {currency}.",
        "نعم، إذا زادت قيمتها عن {limit} {currency}.")],
      ["conflict_of_interest", "anti_bribery"], office_specific=False),

    T("intellectual_property", "LEG",
      "Intellectual Property Policy", "سياسة الملكية الفكرية",
      "This policy states who owns work created during employment.",
      "تحدد هذه السياسة من يملك العمل المُنشأ أثناء التوظيف.",
      [("Work created for {org} as part of your job belongs to {org}.",
        "العمل المُنشأ لصالح {org} كجزء من وظيفتك يملكه {org}."),
       ("Using third-party code or content requires a compatible licence, checked with Legal.",
        "يتطلب استخدام كود أو محتوى طرف ثالث ترخيصًا متوافقًا يُراجَع مع الإدارة القانونية."),
       ("Personal projects on your own time and equipment remain yours, if unrelated to company work.",
        "تبقى المشاريع الشخصية في وقتك ومعداتك ملكًا لك، إذا لم تكن متصلة بعمل الشركة.")],
      {},
      [("Who owns code I write for my job?", "من يملك الكود الذي أكتبه لوظيفتي؟",
        "The company, if it was created as part of your job.",
        "الشركة، إذا أُنشئ كجزء من وظيفتك.")],
      ["confidentiality", "software_licensing"], office_specific=False),

    T("competition_law", "LEG",
      "Competition Law Policy", "سياسة قانون المنافسة",
      "This policy helps employees avoid anti-competitive behaviour.",
      "تساعد هذه السياسة الموظفين على تجنب السلوك المخل بالمنافسة.",
      [("Do not discuss prices, customers or markets with competitors.",
        "لا تناقش الأسعار أو العملاء أو الأسواق مع المنافسين."),
       ("Leave any meeting where competitors start such a discussion, and tell Legal.",
        "غادر أي اجتماع يبدأ فيه المنافسون نقاشًا كهذا، وأبلغ الإدارة القانونية."),
       ("Market information about competitors is gathered only from public sources.",
        "تُجمع المعلومات السوقية عن المنافسين من مصادر عامة فقط.")],
      {},
      [("A competitor wants to talk about pricing - what do I do?",
        "منافس يريد الحديث عن التسعير - ماذا أفعل؟",
        "Do not take part, leave the discussion, and tell Legal.",
        "لا تشارك، غادر النقاش، وأبلغ الإدارة القانونية.")],
      ["confidentiality", "external_communications"], office_specific=False),

    # ---------------- OPS ----------------
    T("client_escalation", "OPS",
      "Client Escalation Policy", "سياسة تصعيد شكاوى العملاء",
      "This policy explains how an unresolved client issue is escalated.",
      "توضح هذه السياسة كيفية تصعيد مشكلة عميل لم تُحل.",
      [("If the first point of contact cannot resolve an issue within {hours} hours, it is escalated to the account manager.",
        "إذا لم تستطع نقطة الاتصال الأولى حل المشكلة خلال {hours} ساعة، تُصعَّد إلى مدير الحساب."),
       ("Issues with possible financial loss or a public complaint are flagged to the department head within {flag} hours.",
        "تُرفع المشكلات ذات الخسارة المالية المحتملة أو الشكوى العلنية إلى رئيس القسم خلال {flag} ساعة."),
       ("Every escalation is logged in the CRM with a timeline of actions.",
        "يُسجَّل كل تصعيد في نظام إدارة علاقات العملاء مع جدول زمني للإجراءات.")],
      {"hours": [24, 48], "flag": [2, 4]},
      [("A client problem is not fixed within {hours} hours - what happens next?",
        "مشكلة عميل لم تُحل خلال {hours} ساعة - ماذا يحدث؟",
        "It is escalated to the account manager.", "تُصعَّد إلى مدير الحساب.")],
      ["incident_management", "service_levels"], office_specific=False),

    T("incident_management", "OPS",
      "Incident Management Policy", "سياسة إدارة الحوادث",
      "This policy explains how service incidents are handled.",
      "توضح هذه السياسة كيفية التعامل مع حوادث الخدمة.",
      [("Incidents are logged with a severity from 1 (critical) to 4 (low).",
        "تُسجَّل الحوادث بمستوى خطورة من 1 (حرج) إلى 4 (منخفض)."),
       ("Severity 1 incidents get an incident commander and updates every {update} minutes.",
        "تحصل حوادث المستوى 1 على قائد حادث وتحديثات كل {update} دقيقة."),
       ("A written post-incident review is completed within {review} working days.",
        "تُكمل مراجعة مكتوبة بعد الحادث خلال {review} أيام عمل.")],
      {"update": [15, 30], "review": [3, 5]},
      [("How often are updates given during a critical incident?",
        "كم مرة تُقدَّم التحديثات أثناء حادث حرج؟",
        "Every {update} minutes.", "كل {update} دقيقة.")],
      ["client_escalation", "change_management", "on_call"], office_specific=False),

    T("change_management", "OPS",
      "Change Management Policy", "سياسة إدارة التغيير",
      "This policy explains how changes to live systems are approved.",
      "توضح هذه السياسة كيفية اعتماد التغييرات على الأنظمة الحية.",
      [("Every change to a live system is recorded with a plan, a test and a rollback step.",
        "يُسجَّل كل تغيير على نظام حي مع خطة واختبار وخطوة تراجع."),
       ("Standard low-risk changes are pre-approved; other changes go to the change board.",
        "التغييرات القياسية منخفضة المخاطر معتمدة مسبقًا، وتُحال البقية إلى لجنة التغيير."),
       ("Changes are not deployed on {freeze_day} or during a change freeze.",
        "لا تُنشر التغييرات يوم {freeze_day} أو أثناء تجميد التغيير.")],
      {"freeze_day": ["Thursday", "Friday"]},
      [("Can I deploy a change on Friday?", "هل يمكنني نشر تغيير يوم الجمعة؟",
        "Not on {freeze_day} or during a change freeze.", "ليس يوم {freeze_day} أو أثناء تجميد التغيير.")],
      ["incident_management", "business_continuity"], office_specific=False),

    T("business_continuity", "OPS",
      "Business Continuity Policy", "سياسة استمرارية الأعمال",
      "This policy covers keeping key services running during a disruption.",
      "تغطي هذه السياسة إبقاء الخدمات الأساسية تعمل أثناء الاضطراب.",
      [("Each team has a continuity plan naming its critical services and recovery time objective.",
        "لكل فريق خطة استمرارية تسمّي خدماته الحرجة وهدف زمن التعافي."),
       ("Continuity plans are tested at least once a year.",
        "تُختبر خطط الاستمرارية مرة واحدة سنويًا على الأقل."),
       ("During a major disruption, the crisis team leads and communicates every {update} hour(s).",
        "أثناء اضطراب كبير، يقود فريق الأزمات ويتواصل كل {update} ساعة.")],
      {"update": [1, 2, 4]},
      [("How often are continuity plans tested?", "كم مرة تُختبر خطط الاستمرارية؟",
        "At least once a year.", "مرة واحدة سنويًا على الأقل.")],
      ["change_management", "incident_management"], office_specific=False),

    T("service_levels", "OPS",
      "Service Level Policy", "سياسة مستويات الخدمة",
      "This policy states the response and resolution targets for client-facing services.",
      "تحدد هذه السياسة أهداف الاستجابة والحل للخدمات الموجهة للعملاء.",
      [("First response to a client request is within {response} business hours.",
        "أول رد على طلب العميل خلال {response} ساعات عمل."),
       ("Standard requests are resolved within {resolve} business days.",
        "تُحل الطلبات القياسية خلال {resolve} أيام عمل."),
       ("Service levels are reviewed with each client every quarter.",
        "تُراجَع مستويات الخدمة مع كل عميل كل ربع سنة.")],
      {"response": [2, 4, 8], "resolve": [2, 3, 5]},
      [("How quickly do we respond to a client request?", "بأي سرعة نرد على طلب عميل؟",
        "Within {response} business hours.", "خلال {response} ساعات عمل.")],
      ["client_escalation", "incident_management"], office_specific=False),

    T("on_call", "OPS",
      "On-Call Policy", "سياسة المناوبة",
      "This policy covers out-of-hours on-call duty.",
      "تغطي هذه السياسة مناوبة الطوارئ خارج ساعات العمل.",
      [("On-call runs in {rotation}-week rotations; the schedule is published a month ahead.",
        "تعمل المناوبة بدورات مدتها {rotation} أسبوع، ويُنشر الجدول قبل شهر."),
       ("An on-call allowance of {allowance} {currency} per week is paid, plus overtime for call-outs.",
        "يُدفع بدل مناوبة قدره {allowance} {currency} أسبوعيًا، إضافة إلى العمل الإضافي للاستدعاءات."),
       ("The on-call engineer acknowledges an alert within {ack} minutes.",
        "يستجيب مهندس المناوبة للتنبيه خلال {ack} دقيقة.")],
      {"rotation": [1, 2], "allowance": [300, 500, 800], "currency": ["SAR", "AED", "EGP"], "ack": [10, 15, 30]},
      [("How much is the on-call allowance?", "كم بدل المناوبة؟",
        "{allowance} {currency} per week.", "{allowance} {currency} أسبوعيًا.")],
      ["overtime", "incident_management"], office_specific=False),

    T("knowledge_sharing", "OPS",
      "Knowledge Sharing Policy", "سياسة مشاركة المعرفة",
      "This policy encourages writing down and sharing how things are done.",
      "تشجع هذه السياسة على تدوين ومشاركة طريقة أداء الأمور.",
      [("Every recurring task should have a short written procedure in the shared knowledge base.",
        "لكل مهمة متكررة إجراء مكتوب قصير في قاعدة المعرفة المشتركة."),
       ("Post-incident reviews and design decisions are written up and shared, not kept private.",
        "تُكتب مراجعات ما بعد الحوادث وقرارات التصميم وتُشارك، ولا تبقى خاصة."),
       ("Pages unreviewed for {stale} months are flagged for an owner to update or archive.",
        "تُوسم الصفحات غير المراجعة منذ {stale} أشهر ليقوم مالكها بتحديثها أو أرشفتها.")],
      {"stale": [6, 12]},
      [("Where do I document a recurring task?", "أين أوثّق مهمة متكررة؟",
        "As a short procedure in the shared knowledge base.",
        "كإجراء قصير في قاعدة المعرفة المشتركة.")],
      ["incident_management", "onboarding"], office_specific=False),

    # ---------------- MKT ----------------
    T("social_media", "MKT",
      "Social Media Policy", "سياسة وسائل التواصل الاجتماعي",
      "This policy covers talking about work on personal social media.",
      "تغطي هذه السياسة الحديث عن العمل على وسائل التواصل الشخصية.",
      [("You may discuss your work on personal accounts, but not confidential or unreleased information.",
        "يمكنك مناقشة عملك على حساباتك الشخصية، لكن ليس المعلومات السرية أو غير المعلنة."),
       ("Make clear that opinions are your own, not an official position of {org}.",
        "وضّح أن الآراء آراؤك الشخصية وليست موقفًا رسميًا لـ {org}."),
       ("Only the {owner} team posts on official {org} accounts.",
        "فريق {owner} فقط ينشر على حسابات {org} الرسمية.")],
      {},
      [("Who can post on the company's official social media?", "من يمكنه النشر على حسابات الشركة الرسمية؟",
        "Only the {owner} team.", "فريق {owner} فقط.")],
      ["external_communications", "confidentiality", "brand_usage"], office_specific=False),

    T("external_communications", "MKT",
      "External Communications Policy", "سياسة الاتصالات الخارجية",
      "This policy covers speaking publicly on behalf of the company.",
      "تغطي هذه السياسة التحدث علنًا باسم الشركة.",
      [("Only authorised spokespeople speak to the media or at public events on behalf of {org}.",
        "المتحدثون المفوضون فقط يتحدثون إلى الإعلام أو في الفعاليات العامة باسم {org}."),
       ("Refer media enquiries to {owner} and do not comment yourself.",
        "أحل استفسارات الإعلام إلى {owner} ولا تعلّق بنفسك."),
       ("External talks and articles about company work are reviewed by {owner} before release.",
        "تُراجَع المحاضرات والمقالات الخارجية عن عمل الشركة من {owner} قبل النشر.")],
      {},
      [("A journalist contacted me about the company - what do I do?",
        "تواصل معي صحفي بخصوص الشركة - ماذا أفعل؟",
        "Refer them to {owner} and do not comment.", "أحله إلى {owner} ولا تعلّق.")],
      ["social_media", "press_enquiries", "competition_law"], office_specific=False),

    T("brand_usage", "MKT",
      "Brand Usage Policy", "سياسة استخدام العلامة التجارية",
      "This policy covers correct use of the company logo and name.",
      "تغطي هذه السياسة الاستخدام الصحيح لشعار الشركة واسمها.",
      [("Use only the current logo files from the brand portal; do not stretch, recolour or add effects.",
        "استخدم ملفات الشعار الحالية من بوابة العلامة فقط، ولا تمدّه أو تغيّر لونه أو تضف مؤثرات."),
       ("Partners and suppliers need written permission to use the {org} name or logo.",
        "يحتاج الشركاء والموردون إلى إذن كتابي لاستخدام اسم {org} أو شعارها."),
       ("Templates for slides, documents and email signatures are in the brand portal.",
        "قوالب الشرائح والمستندات وتواقيع البريد موجودة في بوابة العلامة.")],
      {},
      [("Where do I get the company logo?", "من أين أحصل على شعار الشركة؟",
        "From the brand portal - use the current files only.",
        "من بوابة العلامة - استخدم الملفات الحالية فقط.")],
      ["social_media", "external_communications"], office_specific=False),

    T("press_enquiries", "MKT",
      "Press Enquiries Policy", "سياسة استفسارات الصحافة",
      "This policy explains what to do when a journalist makes contact.",
      "توضح هذه السياسة ما يجب فعله عند تواصل صحفي.",
      [("Do not confirm, deny or comment on anything, even off the record.",
        "لا تؤكد أو تنفِ أو تعلّق على أي شيء، حتى بشكل غير رسمي."),
       ("Take the journalist's name, outlet, question and deadline, and pass them to {owner} the same day.",
        "خذ اسم الصحفي والجهة والسؤال والموعد النهائي، وسلّمها إلى {owner} في اليوم نفسه."),
       ("{owner} coordinates a single approved response.",
        "ينسّق {owner} ردًا واحدًا معتمدًا.")],
      {},
      [("Can I speak off the record to a reporter?", "هل يمكنني التحدث بشكل غير رسمي لصحفي؟",
        "No. Do not comment at all; pass it to {owner}.", "لا. لا تعلّق إطلاقًا، وسلّمها إلى {owner}.")],
      ["external_communications"], office_specific=False),

    T("event_approval", "MKT",
      "Event and Sponsorship Approval Policy", "سياسة اعتماد الفعاليات والرعاية",
      "This policy covers approving company events and sponsorships.",
      "تغطي هذه السياسة اعتماد فعاليات الشركة والرعايات.",
      [("Any event or sponsorship using the {org} name needs {owner} approval before commitments are made.",
        "أي فعالية أو رعاية تستخدم اسم {org} تحتاج موافقة {owner} قبل أي التزامات."),
       ("Requests are submitted at least {lead} weeks ahead with a budget and objective.",
        "تُقدَّم الطلبات قبل {lead} أسابيع على الأقل مع ميزانية وهدف."),
       ("Sponsorships over {value} {currency} also need Finance approval.",
        "تحتاج الرعايات التي تتجاوز {value} {currency} إلى موافقة المالية أيضًا.")],
      {"lead": [4, 6, 8], "value": [25000, 50000], "currency": ["SAR", "AED", "EGP"]},
      [("How early must an event request be submitted?", "قبل كم يجب تقديم طلب الفعالية؟",
        "At least {lead} weeks ahead.", "قبل {lead} أسابيع على الأقل.")],
      ["budget_approval", "brand_usage"], office_specific=False),

    # ---------------- PRC ----------------
    T("purchase_request", "PRC",
      "Purchase Request Policy", "سياسة طلب الشراء",
      "This policy explains how to raise a purchase request.",
      "توضح هذه السياسة كيفية رفع طلب شراء.",
      [("A purchase request states what is needed, why, the budget line and the preferred supplier.",
        "يوضح طلب الشراء ما هو المطلوب ولماذا وبند الميزانية والمورد المفضل."),
       ("Do not commit to a supplier or place an order before the request is approved.",
        "لا تلتزم تجاه مورد أو تضع طلبًا قبل اعتماد الطلب."),
       ("Approved requests become a purchase order sent to the supplier by Procurement.",
        "تتحول الطلبات المعتمدة إلى أمر شراء ترسله المشتريات إلى المورد.")],
      {},
      [("Can I order first and raise the request later?", "هل أطلب أولًا وأرفع الطلب لاحقًا؟",
        "No. The request must be approved before any commitment.",
        "لا. يجب اعتماد الطلب قبل أي التزام.")],
      ["procurement", "budget_approval", "supplier_selection"], office_specific=False),

    T("supplier_selection", "PRC",
      "Supplier Selection Policy", "سياسة اختيار الموردين",
      "This policy explains how suppliers are chosen fairly.",
      "توضح هذه السياسة كيفية اختيار الموردين بإنصاف.",
      [("Selection is based on written criteria: price, quality, delivery, and risk.",
        "يعتمد الاختيار على معايير مكتوبة: السعر والجودة والتسليم والمخاطر."),
       ("For spend over {threshold} {currency}, at least {quotes} suppliers are invited to quote.",
        "للإنفاق الذي يتجاوز {threshold} {currency}، يُدعى {quotes} موردين على الأقل لتقديم عروض."),
       ("A conflict of interest with a supplier is declared and the person steps out of the decision.",
        "يُفصح عن أي تضارب مصالح مع مورد ويتنحى الشخص عن القرار.")],
      {"threshold": [10000, 25000], "quotes": [2, 3], "currency": ["SAR", "AED", "EGP"]},
      [("How many suppliers must quote for a large purchase?",
        "كم موردًا يجب أن يقدم عرضًا لشراء كبير؟",
        "At least {quotes} for spend over {threshold} {currency}.",
        "{quotes} على الأقل للإنفاق فوق {threshold} {currency}.")],
      ["purchase_request", "procurement", "conflict_of_interest"], office_specific=False),

    T("contract_renewal", "PRC",
      "Contract Renewal Policy", "سياسة تجديد العقود",
      "This policy makes sure supplier contracts are reviewed before they renew.",
      "تضمن هذه السياسة مراجعة عقود الموردين قبل تجديدها.",
      [("Procurement reviews every contract {months} months before its renewal date.",
        "تراجع المشتريات كل عقد قبل {months} أشهر من تاريخ تجديده."),
       ("Auto-renewal clauses are avoided; renewals are an active decision.",
        "تُتجنَّب بنود التجديد التلقائي، والتجديد قرار فعّال."),
       ("The review checks price, usage, service quality and whether the need still exists.",
        "تتحقق المراجعة من السعر والاستخدام وجودة الخدمة وما إذا كانت الحاجة قائمة.")],
      {"months": [2, 3, 6]},
      [("When is a supplier contract reviewed before renewal?",
        "متى يُراجَع عقد مورد قبل التجديد؟",
        "{months} months before the renewal date.", "قبل {months} أشهر من تاريخ التجديد.")],
      ["procurement", "invoice_processing"], office_specific=False),

    T("purchase_thresholds", "PRC",
      "Purchase Approval Thresholds", "حدود اعتماد المشتريات",
      "This quick reference lists the approval level needed for each spend amount.",
      "يسرد هذا المرجع السريع مستوى الاعتماد المطلوب لكل مبلغ إنفاق.",
      [("Up to {t1} {currency}: line manager.",
        "حتى {t1} {currency}: المدير المباشر."),
       ("{t1} to {t2} {currency}: department head.",
        "من {t1} إلى {t2} {currency}: رئيس القسم."),
       ("Over {t2} {currency}: Procurement Committee and CFO.",
        "أكثر من {t2} {currency}: لجنة المشتريات والمدير المالي.")],
      {"t1": [5000, 10000], "t2": [50000, 100000], "currency": ["SAR", "AED", "EGP"]},
      [("Who approves a {t1} {currency} purchase?", "من يعتمد شراء بقيمة {t1} {currency}؟",
        "Your line manager, up to {t1} {currency}.", "مديرك المباشر، حتى {t1} {currency}.")],
      ["budget_approval", "procurement", "purchase_request"], office_specific=False),

    # ---------------- HR (added: scale-up batch) ----------------
    T("public_holidays", "HR",
      "Public Holidays Policy", "سياسة العطلات الرسمية",
      "This policy lists paid public holidays and what happens if one falls on a weekend.",
      "توضح هذه السياسة العطلات الرسمية المدفوعة وما يحدث إذا صادفت عطلة نهاية الأسبوع.",
      [("T2 observes {count} paid public holidays a year, published on the HR calendar each January.",
        "تمنح {org} {count} عطلة رسمية مدفوعة سنويًا، تُنشر في تقويم الموارد البشرية كل يناير."),
       ("If a holiday falls on a weekend, a floating day is given instead, to be used within {window} months.",
        "إذا صادفت العطلة نهاية الأسبوع، يُمنح يوم بديل يُستخدم خلال {window} أشهر."),
       ("Employees required to work on a public holiday receive a replacement day plus their normal pay.",
        "الموظف الذي يعمل في عطلة رسمية يحصل على يوم بديل بالإضافة إلى أجره المعتاد.")],
      {"count": [10, 12, 13], "window": [2, 3]},
      [("How many public holidays do we get?", "كم عدد العطلات الرسمية التي نحصل عليها؟",
        "{count} paid public holidays a year.", "{count} عطلة رسمية مدفوعة سنويًا."),
       ("What if a holiday falls on a weekend?", "ماذا لو صادفت العطلة نهاية الأسبوع؟",
        "You get a floating day instead, usable within {window} months.",
        "تحصل على يوم بديل يُستخدم خلال {window} أشهر.")],
      ["annual_leave", "flexible_hours"], office_specific=False),

    T("employee_referral", "HR",
      "Employee Referral Program", "برنامج ترشيح الموظفين",
      "This policy explains the bonus for referring a candidate who is hired.",
      "توضح هذه السياسة المكافأة عند ترشيح مرشح يتم توظيفه.",
      [("A successful referral earns the employee a {bonus} AED bonus, paid after the new hire completes probation.",
        "يحصل الموظف على مكافأة {bonus} درهمًا عند نجاح الترشيح، تُصرف بعد إتمام الموظف الجديد لفترة التجربة."),
       ("The role must be advertised internally; referrals for roles at grade M3 and above are not eligible.",
        "يجب أن تكون الوظيفة معلنة داخليًا؛ لا تُقبل الترشيحات للوظائف من درجة M3 فأعلى."),
       ("An employee may not refer a close relative.",
        "لا يجوز للموظف ترشيح أحد أقاربه المقربين.")],
      {"bonus": [2000, 3000, 5000]},
      [("How much is the referral bonus?", "كم قيمة مكافأة الترشيح؟",
        "{bonus} AED, paid after the new hire completes probation.",
        "{bonus} درهمًا، تُصرف بعد إتمام الموظف الجديد لفترة التجربة."),
       ("Can I refer a family member?", "هل يمكنني ترشيح أحد أفراد عائلتي؟",
        "No, close relatives are not eligible.", "لا، لا يجوز ترشيح الأقارب المقربين.")],
      ["onboarding", "code_of_conduct"], office_specific=False),

    T("sabbatical_leave", "HR",
      "Sabbatical Leave Policy", "سياسة الإجازة التفرغية",
      "This policy covers extended unpaid leave for long-serving employees.",
      "تغطي هذه السياسة الإجازة الممتدة غير المدفوعة للموظفين ذوي الخدمة الطويلة.",
      [("Employees with {years}+ years of service may request up to {months} months of unpaid sabbatical leave.",
        "يجوز للموظفين ذوي {years} سنوات خدمة فأكثر طلب إجازة تفرغية غير مدفوعة تصل إلى {months} أشهر."),
       ("The request needs department head and HR approval at least 3 months in advance.",
        "يحتاج الطلب إلى موافقة رئيس القسم والموارد البشرية قبل 3 أشهر على الأقل."),
       ("Job protection is not guaranteed; the employee returns to an available role of similar level.",
        "لا تُضمن حماية الوظيفة؛ يعود الموظف إلى دور متاح من مستوى مماثل.")],
      {"years": [3, 5], "months": [2, 3, 6]},
      [("How many years of service do I need for a sabbatical?", "كم سنة خدمة أحتاج للإجازة التفرغية؟",
        "At least {years} years.", "{years} سنوات على الأقل."),
       ("Is my job guaranteed when I return?", "هل وظيفتي مضمونة عند العودة؟",
        "No, but you return to an available role of similar level.",
        "لا، لكنك تعود إلى دور متاح من مستوى مماثل.")],
      ["annual_leave", "unpaid_leave", "resignation_notice"], office_specific=False),

    T("health_insurance", "HR",
      "Health Insurance Policy", "سياسة التأمين الصحي",
      "This policy explains medical insurance coverage for employees and dependents.",
      "توضح هذه السياسة تغطية التأمين الصحي للموظفين ومعاليهم.",
      [("Coverage starts on the employee's first working day; no waiting period applies.",
        "تبدأ التغطية في أول يوم عمل للموظف؛ لا تُطبَّق فترة انتظار."),
       ("Employees may add up to {dependents} dependents, with the premium shared as per the HR rate card.",
        "يجوز للموظف إضافة حتى {dependents} من المعالين، مع تقاسم القسط وفق جدول أسعار الموارد البشرية."),
       ("Claims are submitted through the insurer's app within {days} days of treatment.",
        "تُقدَّم المطالبات عبر تطبيق شركة التأمين خلال {days} يومًا من العلاج.")],
      {"dependents": [3, 4, 5], "days": [30, 60, 90]},
      [("When does my health insurance start?", "متى يبدأ تأميني الصحي؟",
        "On your first working day, no waiting period.", "في أول يوم عمل، دون فترة انتظار."),
       ("How many dependents can I add?", "كم عدد المعالين الذين يمكنني إضافتهم؟",
        "Up to {dependents}.", "حتى {dependents}.")],
      ["onboarding", "payroll_schedule"], office_specific=False),

    T("tuition_reimbursement", "HR",
      "Tuition Reimbursement Policy", "سياسة سداد الرسوم الدراسية",
      "This policy covers reimbursement for job-related further education.",
      "تغطي هذه السياسة سداد تكاليف التعليم المرتبط بالوظيفة.",
      [("Up to {amount} AED per year is reimbursed for approved, job-related courses with a passing grade.",
        "يُسدَّد ما يصل إلى {amount} درهم سنويًا مقابل دورات معتمدة ومرتبطة بالوظيفة بدرجة نجاح."),
       ("The course must be approved by the manager before it starts; retroactive claims are not accepted.",
        "يجب اعتماد الدورة من المدير قبل بدئها؛ لا تُقبل المطالبات بأثر رجعي."),
       ("The employee agrees to stay {months} months after completion or repay the amount pro-rata.",
        "يوافق الموظف على البقاء {months} أشهر بعد إتمام الدورة أو سداد المبلغ بالتناسب.")],
      {"amount": [5000, 8000, 10000], "months": [6, 12]},
      [("How much tuition is reimbursed per year?", "كم تُسدَّد الرسوم الدراسية سنويًا؟",
        "Up to {amount} AED for approved courses.", "حتى {amount} درهم للدورات المعتمدة."),
       ("Do I need approval before starting the course?", "هل أحتاج موافقة قبل بدء الدورة؟",
        "Yes, retroactive claims are not accepted.", "نعم، لا تُقبل المطالبات بأثر رجعي.")],
      ["training_development", "resignation_notice"], office_specific=False),

    T("company_car", "HR",
      "Company Car Policy", "سياسة سيارة الشركة",
      "This policy covers eligibility for a company car and its personal use.",
      "تغطي هذه السياسة أهلية الحصول على سيارة شركة واستخدامها الشخصي.",
      [("Employees at grade {grade} and above are eligible for a company car or a monthly car allowance.",
        "يستحق الموظفون من درجة {grade} فأعلى سيارة شركة أو بدل سيارة شهري."),
       ("Personal use is allowed; fuel for personal use beyond {km} km a month is the employee's own cost.",
        "يُسمح بالاستخدام الشخصي؛ وقود الاستخدام الشخصي الذي يتجاوز {km} كم شهريًا على نفقة الموظف."),
       ("The vehicle must be serviced on schedule and any accident reported within 24 hours.",
        "يجب صيانة السيارة حسب الجدول والإبلاغ عن أي حادث خلال 24 ساعة.")],
      {"grade": ["M2", "M3"], "km": [300, 500]},
      [("Who is eligible for a company car?", "من يستحق سيارة الشركة؟",
        "Employees at grade {grade} and above.", "الموظفون من درجة {grade} فأعلى."),
       ("Is personal use of the car allowed?", "هل يُسمح بالاستخدام الشخصي للسيارة؟",
        "Yes, within a {km} km/month fuel allowance.", "نعم، ضمن بدل وقود {km} كم شهريًا.")],
      ["expense_reimbursement", "corporate_card"]),

    T("employee_assistance_program", "HR",
      "Employee Assistance Program", "برنامج مساعدة الموظفين",
      "This policy covers free, confidential counselling for employees.",
      "تغطي هذه السياسة الاستشارات المجانية والسرية للموظفين.",
      [("Employees and their immediate family may access up to {sessions} free counselling sessions a year.",
        "يمكن للموظف وأسرته المباشرة الحصول على ما يصل إلى {sessions} جلسات استشارية مجانية سنويًا."),
       ("Sessions are confidential; HR and managers are not told who used the service.",
        "الجلسات سرية؛ لا تُبلَّغ الموارد البشرية أو المدير بهوية من استخدم الخدمة."),
       ("The service is reached through the EAP hotline, available 24/7.",
        "يتم الوصول إلى الخدمة عبر الخط الساخن لبرنامج المساعدة، المتاح على مدار الساعة.")],
      {"sessions": [4, 6, 8]},
      [("How many free counselling sessions do I get?", "كم عدد جلسات الاستشارة المجانية؟",
        "Up to {sessions} a year, for you and your immediate family.",
        "حتى {sessions} جلسات سنويًا، لك ولأسرتك المباشرة."),
       ("Will my manager know I used the EAP?", "هل سيعرف مديري أنني استخدمت البرنامج؟",
        "No, sessions are confidential.", "لا، الجلسات سرية.")],
      ["health_insurance", "anti_harassment"], office_specific=False),

    T("jury_duty_leave", "HR",
      "Jury Duty and Civic Leave", "إجازة الواجب المدني",
      "This policy covers paid leave for jury duty or a court summons.",
      "تغطي هذه السياسة الإجازة المدفوعة للواجب المدني أو استدعاء المحكمة.",
      [("Employees summoned for jury duty or as a witness receive up to {days} paid days.",
        "يحصل الموظف المستدعى للواجب المدني أو كشاهد على ما يصل إلى {days} أيام مدفوعة."),
       ("A copy of the official summons must be given to HR as soon as it is received.",
        "يجب تسليم نسخة من الاستدعاء الرسمي للموارد البشرية فور استلامه."),
       ("Any fee paid by the court for attendance belongs to the employee, not the company.",
        "أي مبلغ تدفعه المحكمة مقابل الحضور يخص الموظف وليس الشركة.")],
      {"days": [5, 10]},
      [("Is jury duty leave paid?", "هل إجازة الواجب المدني مدفوعة؟",
        "Yes, up to {days} paid days.", "نعم، حتى {days} أيام مدفوعة."),
       ("Do I need to show proof of the summons?", "هل أحتاج إثبات الاستدعاء؟",
        "Yes, give HR a copy as soon as you receive it.",
        "نعم، سلّم نسخة للموارد البشرية فور استلامها.")],
      ["annual_leave", "unpaid_leave"], office_specific=False),

    # ---------------- IT (added: scale-up batch) ----------------
    T("mobile_device_management", "IT",
      "Mobile Device Management Policy", "سياسة إدارة الأجهزة المحمولة",
      "This policy covers enrolling phones and tablets that access company email or data.",
      "تغطي هذه السياسة تسجيل الهواتف والأجهزة اللوحية التي تصل إلى بريد الشركة أو بياناتها.",
      [("Any device used for company email must be enrolled in the mobile device management (MDM) system.",
        "يجب تسجيل أي جهاز يُستخدم لبريد الشركة في نظام إدارة الأجهزة المحمولة."),
       ("A lost or stolen enrolled device must be reported within {hours} hours so it can be remotely wiped.",
        "يجب الإبلاغ عن أي جهاز مسجَّل مفقود أو مسروق خلال {hours} ساعة ليتم مسحه عن بُعد."),
       ("Leaving the company or removing the device from MDM erases company data from it, not personal data.",
        "ترك الشركة أو إزالة الجهاز من النظام يمسح بيانات الشركة منه، وليس البيانات الشخصية.")],
      {"hours": [2, 4, 24]},
      [("Do I have to enrol my phone to get company email?", "هل يجب تسجيل هاتفي للحصول على بريد الشركة؟",
        "Yes, any device used for company email must be enrolled.",
        "نعم، يجب تسجيل أي جهاز يُستخدم لبريد الشركة."),
       ("What happens if I lose my enrolled phone?", "ماذا يحدث إذا فقدت هاتفي المسجَّل؟",
        "Report it within {hours} hours so it can be remotely wiped.",
        "أبلغ خلال {hours} ساعة ليتم مسحه عن بُعد.")],
      ["email_usage", "byod", "security_incident"], office_specific=False),

    T("ai_usage", "IT",
      "Generative AI Usage Policy", "سياسة استخدام الذكاء الاصطناعي التوليدي",
      "This policy covers using AI chat and coding tools for work.",
      "تغطي هذه السياسة استخدام أدوات الدردشة والبرمجة بالذكاء الاصطناعي في العمل.",
      [("Only AI tools on the IT-approved list may be used; a new tool needs IT review before use.",
        "لا يجوز استخدام سوى أدوات الذكاء الاصطناعي المعتمدة من تقنية المعلومات؛ الأداة الجديدة تحتاج مراجعة تقنية المعلومات قبل الاستخدام."),
       ("Client data, personal data and unreleased company information must never be entered into a public AI tool.",
        "يُمنع إدخال بيانات العملاء أو البيانات الشخصية أو معلومات الشركة غير المعلنة في أي أداة ذكاء اصطناعي عامة."),
       ("AI-generated content used externally must be reviewed by a person before it is sent or published.",
        "يجب مراجعة أي محتوى بالذكاء الاصطناعي يُستخدم خارجيًا من قِبل شخص قبل إرساله أو نشره.")],
      {},
      [("Can I paste client data into a public AI chatbot?", "هل يمكنني لصق بيانات العميل في روبوت محادثة عام؟",
        "No, client and personal data must never be entered into a public AI tool.",
        "لا، يُمنع إدخال بيانات العملاء أو البيانات الشخصية في أي أداة ذكاء اصطناعي عامة."),
       ("Can I use any AI tool I find useful?", "هل يمكنني استخدام أي أداة ذكاء اصطناعي أجدها مفيدة؟",
        "No, only tools on the IT-approved list; new tools need IT review first.",
        "لا، فقط الأدوات المعتمدة من تقنية المعلومات؛ الأداة الجديدة تحتاج مراجعة أولاً.")],
      ["data_privacy", "acceptable_use", "confidentiality"], office_specific=False),

    T("usb_removable_media", "IT",
      "Removable Media Policy", "سياسة وسائط التخزين القابلة للإزالة",
      "This policy covers using USB drives and other removable storage.",
      "تغطي هذه السياسة استخدام أقراص USB ووسائط التخزين القابلة للإزالة الأخرى.",
      [("Company data may only be copied to removable media that is encrypted and IT-issued.",
        "لا يجوز نسخ بيانات الشركة إلا إلى وسائط قابلة للإزالة مشفَّرة وصادرة من تقنية المعلومات."),
       ("Personal USB drives must not be connected to a company device.",
        "يُمنع توصيل أقراص USB الشخصية بأجهزة الشركة."),
       ("Lost removable media containing company data is reported to IT within {hours} hours.",
        "يُبلَّغ عن أي وسيط تخزين مفقود يحتوي بيانات الشركة إلى تقنية المعلومات خلال {hours} ساعة.")],
      {"hours": [2, 4]},
      [("Can I use my personal USB drive at work?", "هل يمكنني استخدام قرص USB الشخصي في العمل؟",
        "No, personal USB drives must not be connected to a company device.",
        "لا، يُمنع توصيل أقراص USB الشخصية بأجهزة الشركة."),
       ("What kind of removable media is allowed for company data?", "ما نوع وسائط التخزين المسموح بها لبيانات الشركة؟",
        "Only encrypted, IT-issued removable media.", "فقط الوسائط المشفَّرة والصادرة من تقنية المعلومات.")],
      ["data_backup", "encryption", "security_incident"], office_specific=False),

    T("guest_wifi", "IT",
      "Guest Wi-Fi Policy", "سياسة شبكة الضيوف اللاسلكية",
      "This policy covers internet access for visitors.",
      "تغطي هذه السياسة وصول الزوار إلى الإنترنت.",
      [("Guests are given access to a separate guest network only; it cannot reach internal systems.",
        "يُمنح الزوار وصولاً إلى شبكة ضيوف منفصلة فقط؛ لا يمكنها الوصول إلى الأنظمة الداخلية."),
       ("The guest Wi-Fi password is changed every {days} days and is available from reception.",
        "تُغيَّر كلمة مرور شبكة الضيوف كل {days} يومًا وتُتاح من الاستقبال."),
       ("A staff member sponsoring a guest is responsible for that guest's network use.",
        "الموظف الذي يستضيف زائرًا مسؤول عن استخدام ذلك الزائر للشبكة.")],
      {"days": [7, 14, 30]},
      [("Can guests access our internal systems over the guest Wi-Fi?", "هل يمكن للزوار الوصول للأنظمة الداخلية عبر شبكة الضيوف؟",
        "No, the guest network cannot reach internal systems.",
        "لا، لا يمكن لشبكة الضيوف الوصول إلى الأنظمة الداخلية."),
       ("How often does the guest Wi-Fi password change?", "كم مرة تتغير كلمة مرور شبكة الضيوف؟",
        "Every {days} days.", "كل {days} يومًا.")],
      ["visitor_management", "vpn_remote_access"]),

    T("password_reset_self_service", "IT",
      "Password Self-Service Reset", "إعادة تعيين كلمة المرور الذاتية",
      "This policy covers resetting a forgotten password without contacting the help desk.",
      "تغطي هذه السياسة إعادة تعيين كلمة مرور منسية دون التواصل مع مكتب المساعدة.",
      [("Employees can reset a forgotten password through the self-service portal after verifying identity.",
        "يمكن للموظف إعادة تعيين كلمة مرور منسية عبر بوابة الخدمة الذاتية بعد التحقق من الهوية."),
       ("The account locks after {attempts} failed sign-in attempts and unlocks after {minutes} minutes.",
        "يُقفل الحساب بعد {attempts} محاولات دخول فاشلة ويُفتح بعد {minutes} دقيقة."),
       ("If self-service verification fails, contact the service desk with a photo ID.",
        "إذا فشل التحقق الذاتي، تواصل مع مكتب الخدمة مع إثبات هوية بصورة.")],
      {"attempts": [3, 5], "minutes": [15, 30]},
      [("How many failed logins lock my account?", "كم محاولة دخول فاشلة تقفل حسابي؟",
        "{attempts} attempts, unlocking after {minutes} minutes.",
        "{attempts} محاولات، ويُفتح بعد {minutes} دقيقة."),
       ("Can I reset my password myself?", "هل يمكنني إعادة تعيين كلمة المرور بنفسي؟",
        "Yes, through the self-service portal after verifying your identity.",
        "نعم، عبر بوابة الخدمة الذاتية بعد التحقق من الهوية.")],
      ["password_policy", "help_desk_sla"], office_specific=False),

    # ---------------- Finance (added: scale-up batch) ----------------
    T("travel_advance", "FIN",
      "Travel Cash Advance Policy", "سياسة السلفة النقدية للسفر",
      "This policy covers requesting a cash advance before a business trip.",
      "تغطي هذه السياسة طلب سلفة نقدية قبل رحلة عمل.",
      [("An advance of up to {amount} may be requested for trip costs not covered by the corporate card.",
        "يمكن طلب سلفة تصل إلى {amount} لتغطية تكاليف الرحلة غير المشمولة ببطاقة الشركة."),
       ("The advance is requested at least {days} days before travel through the finance portal.",
        "تُطلب السلفة قبل {days} أيام على الأقل من السفر عبر بوابة المالية."),
       ("The advance is reconciled against receipts within 14 days of returning; any unused amount is returned.",
        "تُسوَّى السلفة مقابل الإيصالات خلال 14 يومًا من العودة، ويُعاد أي مبلغ غير مستخدم.")],
      {"amount": [2000, 3000, 5000], "days": [3, 5]},
      [("How much travel advance can I request?", "كم سلفة سفر يمكنني طلبها؟",
        "Up to {amount}, for costs not covered by the corporate card.",
        "حتى {amount}، لتغطية التكاليف غير المشمولة ببطاقة الشركة."),
       ("When must I reconcile my travel advance?", "متى يجب تسوية سلفة السفر؟",
        "Within 14 days of returning from the trip.", "خلال 14 يومًا من العودة من الرحلة.")],
      ["travel_booking", "per_diem", "expense_reimbursement"]),

    T("currency_exchange", "FIN",
      "Foreign Currency Expense Policy", "سياسة مصاريف العملات الأجنبية",
      "This policy covers converting foreign-currency expenses for reimbursement.",
      "تغطي هذه السياسة تحويل مصاريف العملات الأجنبية لغرض السداد.",
      [("Foreign-currency receipts are converted using the exchange rate on the date of the transaction.",
        "تُحوَّل الإيصالات بالعملات الأجنبية باستخدام سعر الصرف في تاريخ المعاملة."),
       ("The rate source is the corporate card statement rate, or the central bank rate if paid in cash.",
        "مصدر السعر هو سعر كشف حساب بطاقة الشركة، أو سعر البنك المركزي إذا دُفع نقدًا."),
       ("Reimbursement is always paid in the employee's home payroll currency.",
        "يُصرف السداد دائمًا بعملة الرواتب الخاصة بالموظف.")],
      {},
      [("What exchange rate is used for a foreign receipt?", "ما سعر الصرف المستخدم لإيصال بعملة أجنبية؟",
        "The rate on the date of the transaction.", "السعر في تاريخ المعاملة."),
       ("What currency will I be reimbursed in?", "بأي عملة سأحصل على السداد؟",
        "Your home payroll currency.", "عملة الرواتب الخاصة بك.")],
      ["expense_reimbursement", "travel_advance", "corporate_card"], office_specific=False),

    T("bank_account_change", "FIN",
      "Payroll Bank Account Change", "تغيير حساب استلام الراتب",
      "This policy covers updating the bank account salary is paid into.",
      "تغطي هذه السياسة تحديث الحساب البنكي الذي يُصرف إليه الراتب.",
      [("A bank account change is submitted through the HR system with a bank letter or void cheque.",
        "يُقدَّم تغيير الحساب البنكي عبر نظام الموارد البشرية مع خطاب بنكي أو شيك ملغى."),
       ("Changes submitted after the {day} of the month apply from the following month's payroll.",
        "التغييرات المقدَّمة بعد يوم {day} من الشهر تُطبَّق من رواتب الشهر التالي."),
       ("Payroll will call the employee directly to confirm any account change before paying it.",
        "ستتصل جهة الرواتب بالموظف مباشرة لتأكيد أي تغيير في الحساب قبل الصرف إليه.")],
      {"day": [15, 20, 25]},
      [("How do I change my salary bank account?", "كيف أغيّر حساب استلام راتبي؟",
        "Submit it through the HR system with a bank letter or void cheque.",
        "قدّمه عبر نظام الموارد البشرية مع خطاب بنكي أو شيك ملغى."),
       ("Will payroll confirm my account change?", "هل ستؤكد جهة الرواتب تغيير حسابي؟",
        "Yes, by calling you directly before paying it.", "نعم، بالاتصال بك مباشرة قبل الصرف إليه.")],
      ["payroll_schedule", "tax_documents"], office_specific=False),

    T("asset_capitalization", "FIN",
      "Asset Capitalization Policy", "سياسة رسملة الأصول",
      "This policy sets the threshold for treating a purchase as a company asset.",
      "تحدد هذه السياسة الحد الذي يُعامل عنده الشراء كأصل للشركة.",
      [("A purchase of {threshold} AED or more with a useful life over one year is capitalized as an asset.",
        "يُرسمل الشراء بقيمة {threshold} درهم أو أكثر وعمر إنتاجي يتجاوز سنة كأصل."),
       ("Capitalized assets are depreciated over {years} years on a straight-line basis.",
        "تُستهلك الأصول المرسملة على مدى {years} سنوات بطريقة القسط الثابت."),
       ("Finance tags and records every capitalized asset in the fixed asset register.",
        "تُسجِّل المالية وتُرمِّز كل أصل مرسمل في سجل الأصول الثابتة.")],
      {"threshold": [2000, 3000, 5000], "years": [3, 5]},
      [("What purchase amount counts as a capital asset?", "ما المبلغ الذي يُعتبر به الشراء أصلًا رأسماليًا؟",
        "{threshold} AED or more, with a useful life over one year.",
        "{threshold} درهم أو أكثر، بعمر إنتاجي يتجاوز سنة."),
       ("Over how many years is an asset depreciated?", "على مدى كم سنة يُستهلك الأصل؟",
        "{years} years, straight-line.", "{years} سنوات بطريقة القسط الثابت.")],
      ["procurement", "purchase_thresholds"], office_specific=False),

    # ---------------- Facilities (added: scale-up batch) ----------------
    T("desk_booking", "FAC",
      "Hot Desk Booking Policy", "سياسة حجز المكاتب المشتركة",
      "This policy covers booking a shared desk in flexible-seating areas.",
      "تغطي هذه السياسة حجز مكتب مشترك في مناطق الجلوس المرن.",
      [("Desks in shared-seating areas are booked through the office app up to {days} days ahead.",
        "تُحجَز المكاتب في مناطق الجلوس المشترك عبر تطبيق المكتب قبل {days} أيام."),
       ("A desk not checked into within {minutes} minutes of the booking start is released to others.",
        "المكتب الذي لا يُسجَّل الدخول إليه خلال {minutes} دقيقة من بدء الحجز يُطلق لآخرين."),
       ("Personal items are cleared at the end of the day; the desk is shared the next day.",
        "تُزال الأغراض الشخصية في نهاية اليوم؛ يُشارك المكتب في اليوم التالي.")],
      {"days": [5, 7, 14], "minutes": [15, 30]},
      [("How far ahead can I book a hot desk?", "قبل كم يوم يمكنني حجز مكتب مشترك؟",
        "Up to {days} days ahead.", "حتى {days} أيام مسبقًا."),
       ("What happens if I don't check in to my booked desk?", "ماذا يحدث إذا لم أسجل الدخول لمكتبي المحجوز؟",
        "It's released to others after {minutes} minutes.", "يُطلق لآخرين بعد {minutes} دقيقة.")],
      ["meeting_room_booking", "remote_work"]),

    T("ev_charging", "FAC",
      "EV Charging Policy", "سياسة شحن السيارات الكهربائية",
      "This policy covers using electric-vehicle charging points at the office.",
      "تغطي هذه السياسة استخدام نقاط شحن السيارات الكهربائية في المكتب.",
      [("EV charging bays are booked through the office app; a session is limited to {hours} hours.",
        "تُحجَز مواقف الشحن عبر تطبيق المكتب؛ تقتصر الجلسة على {hours} ساعات."),
       ("Charging is free for employees; moving the car promptly once charged is expected.",
        "الشحن مجاني للموظفين؛ يُتوقع نقل السيارة فور اكتمال الشحن."),
       ("Non-EVs must not park in a charging bay, even briefly.",
        "يُمنع وقوف السيارات غير الكهربائية في مواقف الشحن ولو لفترة وجيزة.")],
      {"hours": [2, 3, 4]},
      [("Is EV charging free?", "هل الشحن مجاني؟",
        "Yes, free for employees.", "نعم، مجاني للموظفين."),
       ("How long can I charge for?", "كم مدة يمكنني الشحن؟",
        "Sessions are limited to {hours} hours.", "تقتصر الجلسة على {hours} ساعات.")],
      ["parking", "building_access"]),

    T("smoking_policy", "FAC",
      "Smoking Policy", "سياسة التدخين",
      "This policy covers where smoking is allowed on company premises.",
      "تغطي هذه السياسة الأماكن المسموح فيها بالتدخين في مقار الشركة.",
      [("Smoking, including e-cigarettes, is allowed only in designated outdoor areas.",
        "يُسمح بالتدخين، بما في ذلك السجائر الإلكترونية، فقط في المناطق الخارجية المخصصة."),
       ("Designated areas are at least {meters} metres from any building entrance.",
        "تبعد المناطق المخصصة {meters} مترًا على الأقل عن أي مدخل مبنى."),
       ("Smoking anywhere else on the premises, including company vehicles, is not permitted.",
        "يُمنع التدخين في أي مكان آخر بالمقر، بما في ذلك مركبات الشركة.")],
      {"meters": [10, 15, 20]},
      [("Where can I smoke on site?", "أين يمكنني التدخين في الموقع؟",
        "Only in designated outdoor areas, {meters} metres from entrances.",
        "فقط في المناطق الخارجية المخصصة، على بُعد {meters} مترًا من المداخل."),
       ("Are e-cigarettes treated the same as smoking?", "هل تُعامل السجائر الإلكترونية معاملة التدخين؟",
        "Yes, the same rules apply.", "نعم، تُطبَّق القواعد نفسها.")],
      ["health_safety", "fire_safety"]),

    T("lost_and_found", "FAC",
      "Lost and Found Policy", "سياسة المفقودات",
      "This policy covers items found on company premises.",
      "تغطي هذه السياسة الأغراض التي يُعثر عليها في مقار الشركة.",
      [("Found items are handed to reception and logged with the date and location found.",
        "تُسلَّم الأغراض المعثور عليها إلى الاستقبال وتُسجَّل بتاريخ ومكان العثور."),
       ("Unclaimed items are kept for {days} days, then donated or disposed of.",
        "تُحفظ الأغراض غير المطالب بها لمدة {days} يومًا، ثم تُتبرَّع بها أو يُتخلَّص منها."),
       ("Cash, cards and ID documents are handed to security immediately, not kept at reception.",
        "تُسلَّم النقود والبطاقات ووثائق الهوية إلى الأمن فورًا، ولا تُحفظ في الاستقبال.")],
      {"days": [30, 60, 90]},
      [("How long are lost items kept?", "كم مدة تُحفظ الأغراض المفقودة؟",
        "{days} days, then donated or disposed of.", "{days} يومًا، ثم تُتبرَّع بها أو يُتخلَّص منها."),
       ("What happens if I find a lost wallet?", "ماذا أفعل إذا وجدت محفظة مفقودة؟",
        "Hand cash, cards or ID straight to security, not reception.",
        "سلّم النقود أو البطاقات أو الهوية مباشرة إلى الأمن، وليس الاستقبال.")],
      ["building_access", "visitor_management"]),

    # ---------------- Legal & Compliance (added: scale-up batch) ----------------
    T("data_subject_requests", "LEG",
      "Data Subject Access Request Policy", "سياسة طلبات الوصول لبيانات الأفراد",
      "This policy covers a request from a person to see or delete their personal data.",
      "تغطي هذه السياسة طلب شخص الاطلاع على بياناته الشخصية أو حذفها.",
      [("A data subject request is forwarded to Legal & Compliance the same day it is received.",
        "يُحال طلب صاحب البيانات إلى الشؤون القانونية والامتثال في نفس يوم استلامه."),
       ("The company responds within {days} days, as required by data protection law.",
        "تستجيب الشركة خلال {days} يومًا، وفقًا لما يقتضيه قانون حماية البيانات."),
       ("The requester's identity is verified before any personal data is shared or deleted.",
        "تُتحقَّق هوية مقدم الطلب قبل مشاركة أي بيانات شخصية أو حذفها.")],
      {"days": [30, 45]},
      [("What do I do if a customer asks to see their data?", "ماذا أفعل إذا طلب عميل الاطلاع على بياناته؟",
        "Forward the request to Legal & Compliance the same day.",
        "أحِل الطلب إلى الشؤون القانونية والامتثال في نفس اليوم."),
       ("How long does the company have to respond?", "كم من الوقت لدى الشركة للاستجابة؟",
        "{days} days.", "{days} يومًا.")],
      ["data_privacy", "confidentiality", "records_retention"], office_specific=False),

    T("trademark_usage", "LEG",
      "Trademark and Logo Usage Policy", "سياسة استخدام العلامة التجارية والشعار",
      "This policy covers using the company name, logo and brand assets.",
      "تغطي هذه السياسة استخدام اسم الشركة وشعارها وأصولها التجارية.",
      [("The logo and brand assets may only be used in their approved form, from the brand portal.",
        "لا يجوز استخدام الشعار والأصول التجارية إلا بصيغتها المعتمدة من بوابة العلامة التجارية."),
       ("Any external use - a partner site, an event, a press item - needs Corporate Communications approval.",
        "أي استخدام خارجي - موقع شريك أو فعالية أو مادة صحفية - يحتاج موافقة الاتصال المؤسسي."),
       ("Employees may not register a domain, page or account using the company name without approval.",
        "لا يجوز للموظف تسجيل نطاق أو صفحة أو حساب باسم الشركة دون موافقة.")],
      {},
      [("Can I use the company logo on a partner's website?", "هل يمكنني استخدام شعار الشركة في موقع شريك؟",
        "Only with Corporate Communications approval.", "فقط بموافقة الاتصال المؤسسي."),
       ("Where do I get the approved logo files?", "من أين أحصل على ملفات الشعار المعتمدة؟",
        "From the brand portal.", "من بوابة العلامة التجارية.")],
      ["brand_usage", "external_communications", "intellectual_property"], office_specific=False),

    T("non_compete", "LEG",
      "Non-Compete and Non-Solicit Policy", "سياسة عدم المنافسة وعدم الاستقطاب",
      "This policy explains restrictions that continue after an employee leaves.",
      "توضح هذه السياسة القيود التي تستمر بعد ترك الموظف الشركة.",
      [("Senior roles (grade {grade} and above) may not join a direct competitor for {months} months after leaving.",
        "لا يجوز لشاغلي الوظائف العليا (درجة {grade} فأعلى) الانضمام لمنافس مباشر خلال {months} أشهر بعد المغادرة."),
       ("No employee may solicit T2 clients or staff for {months} months after leaving.",
        "لا يجوز لأي موظف استقطاب عملاء أو موظفي {org} خلال {months} أشهر بعد المغادرة."),
       ("The exact terms are set out in the employee's contract, not this policy alone.",
        "تُحدَّد الشروط الدقيقة في عقد الموظف، وليس في هذه السياسة وحدها.")],
      {"grade": ["M3", "M4"], "months": [6, 12]},
      [("Does the non-compete apply to everyone?", "هل يسري بند عدم المنافسة على الجميع؟",
        "No, only senior roles at grade {grade} and above.", "لا، فقط الوظائف العليا من درجة {grade} فأعلى."),
       ("How long does the non-solicit rule last after I leave?", "كم تستمر قاعدة عدم الاستقطاب بعد المغادرة؟",
        "{months} months.", "{months} أشهر.")],
      ["resignation_notice", "confidentiality", "conflict_of_interest"], office_specific=False),

    # ---------------- Operations (added: scale-up batch) ----------------
    T("customer_feedback", "OPS",
      "Customer Feedback Handling Policy", "سياسة التعامل مع ملاحظات العملاء",
      "This policy covers logging and responding to customer feedback.",
      "تغطي هذه السياسة تسجيل ملاحظات العملاء والرد عليها.",
      [("All customer feedback, positive or negative, is logged in the CRM within {hours} hours of receipt.",
        "تُسجَّل جميع ملاحظات العملاء، الإيجابية والسلبية، في نظام إدارة علاقات العملاء خلال {hours} ساعة من استلامها."),
       ("A negative comment about safety or fraud is escalated to the department head immediately.",
        "تُصعَّد أي ملاحظة سلبية متعلقة بالسلامة أو الاحتيال إلى رئيس القسم فورًا."),
       ("The customer receives an acknowledgement within one business day, even if the fix takes longer.",
        "يتلقى العميل إشعارًا باستلام ملاحظته خلال يوم عمل واحد، حتى لو استغرق الحل وقتًا أطول.")],
      {"hours": [24, 48]},
      [("How fast must customer feedback be logged?", "بأي سرعة يجب تسجيل ملاحظات العملاء؟",
        "Within {hours} hours of receipt.", "خلال {hours} ساعة من استلامها."),
       ("What if the feedback is about a safety issue?", "ماذا لو كانت الملاحظة متعلقة بمسألة سلامة؟",
        "It is escalated to the department head immediately.", "تُصعَّد إلى رئيس القسم فورًا.")],
      ["client_escalation", "incident_management"], office_specific=False),

    T("supplier_performance", "OPS",
      "Supplier Performance Review Policy", "سياسة تقييم أداء الموردين",
      "This policy covers scoring supplier performance over time.",
      "تغطي هذه السياسة تقييم أداء الموردين بمرور الوقت.",
      [("Key suppliers are scored every {months} months on quality, delivery and responsiveness.",
        "يُقيَّم الموردون الرئيسيون كل {months} أشهر على الجودة والتسليم والاستجابة."),
       ("A score below {threshold} out of 100 triggers a formal improvement plan.",
        "يؤدي التقييم الأقل من {threshold} من 100 إلى خطة تحسين رسمية."),
       ("Two consecutive low scores are referred to Procurement for a sourcing review.",
        "يُحال تقييمان منخفضان متتاليان إلى المشتريات لمراجعة مصدر التوريد.")],
      {"months": [3, 6], "threshold": [60, 70]},
      [("How often are suppliers scored?", "كم مرة يُقيَّم الموردون؟",
        "Every {months} months.", "كل {months} أشهر."),
       ("What happens if a supplier scores poorly?", "ماذا يحدث إذا كان تقييم المورد ضعيفًا؟",
        "Below {threshold}/100 triggers a formal improvement plan.",
        "أقل من {threshold} من 100 يؤدي إلى خطة تحسين رسمية.")],
      ["supplier_selection", "vendor_onboarding", "contract_renewal"], office_specific=False),
]

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
FIRST_NAMES = ["Sara", "Omar", "Lina", "Khalid", "Noura", "Yousef", "Huda", "Tariq",
               "Maha", "Faisal", "Rana", "Ali", "Dana", "Hassan", "Reem", "Ziad"]
LAST_NAMES = ["Al-Harbi", "Nasser", "Saleh", "Al-Qahtani", "Habib", "Farouk", "Aziz",
              "Mansour", "Rashid", "Darwish", "Kamal", "Sabbagh", "Younis", "Hariri"]


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:48]


def fill(text: str, values: dict) -> str:
    out = text.replace("{org}", ORG).replace("{domain}", CONTACT_DOMAIN)
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def pick_params(topic: Topic, rng: random.Random, shift: int) -> dict:
    values: dict = {}
    for name, options in topic.params.items():
        if not options:
            continue
        idx = rng.randrange(len(options))
        if all(isinstance(o, (int, float)) for o in options):
            idx = max(0, min(len(options) - 1, idx + shift))
        values[name] = options[idx]
    return values


def owner_name(dept: str, lang: str) -> str:
    return OWNERS[dept][0 if lang == "en" else 1]


@dataclass
class DocSpec:
    topic: Topic
    doc_type: str
    lang: str
    office_key: str
    office_en: str
    office_ar: str
    shift: int
    version_major: int
    superseded_by: str | None = None


def doc_title(spec: DocSpec) -> str:
    t = spec.topic
    en = spec.lang == "en"
    title = t.title_en if en else t.title_ar
    if spec.doc_type != "Policy":
        title = re.sub(r"\s+Policy$", "", title)
        title = re.sub(r"^سياسة\s+", "", title)
        suffix = {"Procedure": (" - Procedure", " - إجراء"),
                  "FAQ": (" - FAQ", " - أسئلة شائعة"),
                  "Quick Reference": (" - Quick Reference", " - مرجع سريع")}[spec.doc_type]
        title += suffix[0] if en else suffix[1]
    if t.office_specific and spec.office_key != "HO":
        title += f" ({spec.office_en})" if en else f" ({spec.office_ar})"
    return title


def make_body(spec: DocSpec, values: dict, doc_id: str, eff_date: date,
              version: str, status: str, related_ids: list[str]) -> str:
    t = spec.topic
    lang = spec.lang
    en = lang == "en"
    title = doc_title(spec)

    dept_full = DEPARTMENTS[t.dept]
    owner = owner_name(t.dept, lang)
    if not t.office_specific or spec.office_key == "HO":
        applies = "All employees" if en else "جميع الموظفين"
        applies_lc = "all employees" if en else "جميع الموظفين"
    else:
        applies = (spec.office_en + " employees") if en else ("موظفي " + spec.office_ar)
        applies_lc = applies

    L: list[str] = []
    lbl = {
        "title": "Title" if en else "العنوان",
        "id": "Document ID" if en else "معرّف الوثيقة",
        "dept": "Department" if en else "القسم",
        "type": "Type" if en else "النوع",
        "ver": "Version" if en else "الإصدار",
        "eff": "Effective date" if en else "تاريخ السريان",
        "status": "Status" if en else "الحالة",
        "owner": "Owner" if en else "الجهة المالكة",
        "applies": "Applies to" if en else "ينطبق على",
        "related": "Related documents" if en else "وثائق ذات صلة",
    }
    st = status if en else ("سارية" if status == "Active" else "ملغاة ومستبدلة")
    dtype = spec.doc_type if en else {
        "Policy": "سياسة", "Procedure": "إجراء", "FAQ": "أسئلة شائعة",
        "Quick Reference": "مرجع سريع"}[spec.doc_type]

    L.append(f"{lbl['title']}: {title}")
    L.append(f"{lbl['id']}: {doc_id}")
    L.append(f"{lbl['dept']}: {dept_full if en else OWNERS[t.dept][1]}")
    L.append(f"{lbl['type']}: {dtype}")
    L.append(f"{lbl['ver']}: {version}")
    L.append(f"{lbl['eff']}: {eff_date.isoformat()}")
    L.append(f"{lbl['status']}: {st}")
    L.append(f"{lbl['owner']}: {owner}")
    L.append(f"{lbl['applies']}: {applies}")
    if related_ids:
        L.append(f"{lbl['related']}: {', '.join(related_ids)}")
    if spec.superseded_by:
        L.append((f"Superseded by: {spec.superseded_by}") if en
                 else f"استُبدلت بـ: {spec.superseded_by}")
    L.append("")

    clauses = [fill(c_en if en else c_ar, {**values, "owner": owner})
               for (c_en, c_ar) in t.clauses]
    purpose = fill(t.purpose_en if en else t.purpose_ar, values)

    if spec.doc_type == "Policy":
        L.append(("1. Purpose" if en else "1. الغرض"))
        L.append(purpose)
        L.append("")
        L.append(("2. Scope" if en else "2. النطاق"))
        L.append((f"This policy applies to {applies_lc}." if en
                  else f"تنطبق هذه السياسة على {applies_lc}."))
        L.append("")
        L.append(("3. Policy" if en else "3. السياسة"))
        for c in clauses:
            L.append(f"- {c}")
        L.append("")
        L.append(("4. Exceptions" if en else "4. الاستثناءات"))
        L.append(("Exceptions need written approval from " + owner + "." if en
                  else "تتطلب الاستثناءات موافقة كتابية من " + owner + "."))
        L.append("")
        L.append(("5. Contact" if en else "5. جهة الاتصال"))
        L.append((f"For questions, contact {t.dept.lower()}@{CONTACT_DOMAIN}." if en
                  else f"للاستفسارات، تواصل مع {t.dept.lower()}@{CONTACT_DOMAIN}."))

    elif spec.doc_type == "Procedure":
        L.append(("Purpose" if en else "الغرض"))
        L.append(purpose)
        L.append("")
        L.append(("Steps" if en else "الخطوات"))
        steps_en = [
            "Check this document and any related policy before you start.",
            "Prepare the information you need.",
            "Submit the request in the correct system.",
            "Wait for the required approval.",
            "Keep a copy of the approval for your records.",
        ]
        steps_ar = [
            "راجع هذه الوثيقة وأي سياسة ذات صلة قبل البدء.",
            "جهّز المعلومات التي تحتاجها.",
            "قدّم الطلب في النظام الصحيح.",
            "انتظر الموافقة المطلوبة.",
            "احتفظ بنسخة من الموافقة لسجلاتك.",
        ]
        for i, s in enumerate(steps_en if en else steps_ar, 1):
            L.append(f"{i}. {s}")
        L.append("")
        L.append(("Rules that apply" if en else "القواعد المطبقة"))
        for c in clauses:
            L.append(f"- {c}")
        L.append("")
        L.append(("Contact" if en else "جهة الاتصال"))
        L.append((f"{owner}: {t.dept.lower()}@{CONTACT_DOMAIN}"))

    elif spec.doc_type == "FAQ":
        L.append((f"Frequently asked questions about {title.lower()}." if en
                  else f"أسئلة شائعة حول {title}."))
        L.append("")
        faqs = t.faq or [(
            "What does this cover?" if en else "ماذا تغطي هذه الوثيقة؟",
            "What does this cover?" if en else "ماذا تغطي هذه الوثيقة؟",
            purpose, purpose)]
        for (q_en, q_ar, a_en, a_ar) in faqs:
            L.append(("Q: " if en else "س: ") + fill(q_en if en else q_ar, values))
            L.append(("A: " if en else "ج: ") + fill(a_en if en else a_ar,
                                                     {**values, "owner": owner}))
            L.append("")
        L.append(("More detail is in the matching policy document." if en
                  else "مزيد من التفاصيل في وثيقة السياسة المقابلة."))

    else:  # Quick Reference
        L.append((f"Quick reference - {title}." if en else f"مرجع سريع - {title}."))
        L.append("")
        L.append(("Key points" if en else "النقاط الرئيسية"))
        for c in clauses:
            L.append(f"- {c}")
        L.append("")
        L.append((f"Owner: {owner}. Full policy: {doc_id.rsplit('-', 1)[0]} policy set." if en
                  else f"الجهة المالكة: {owner}. السياسة الكاملة: مجموعة سياسات {t.dept}."))

    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _interleave(items: list[DocSpec], key) -> list[DocSpec]:
    """Round-robin the items across the values of key(item), so no single
    group (e.g. department) dominates the front of the list."""
    from collections import OrderedDict, deque

    buckets: OrderedDict[str, deque] = OrderedDict()
    for it in items:
        buckets.setdefault(key(it), deque()).append(it)
    out: list[DocSpec] = []
    while any(buckets.values()):
        for b in buckets.values():
            if b:
                out.append(b.popleft())
    return out


def build(out_dir: Path, target: int) -> None:
    (out_dir / "en").mkdir(parents=True, exist_ok=True)
    (out_dir / "ar").mkdir(parents=True, exist_ok=True)

    branch_offices = [o for o in OFFICES if o[0] != "HO"]

    # Tier 1: one Head-Office document per topic x doc-type x language.
    tier1: list[DocSpec] = []
    for t in TOPICS:
        for dt in DOC_TYPES:
            for lang in ("en", "ar"):
                tier1.append(DocSpec(t, dt, lang, "HO", "Head Office",
                                     "المكتب الرئيسي", 0, 3))

    # Tier 2: per-office variants (numbers and scope differ by office).
    tier2: list[DocSpec] = []
    for (okey, oen, oar, _w, shift) in branch_offices:
        for t in TOPICS:
            if not t.office_specific:
                continue
            for dt in ("Policy", "Quick Reference", "Procedure"):
                for lang in ("en", "ar"):
                    tier2.append(DocSpec(t, dt, lang, okey, oen, oar, shift, 2))

    # Balance both tiers across departments so HR does not dominate the corpus.
    specs = _interleave(tier1, lambda s: s.topic.dept) \
        + _interleave(tier2, lambda s: s.topic.dept)
    specs = specs[:target]

    # assign sequential IDs per department
    dept_counter: dict[str, int] = {d: 0 for d in DEPARTMENTS}
    rows = []
    base_date = date(2023, 1, 1)

    # first pass: allocate ids so we can cross-link
    for s in specs:
        dept_counter[s.topic.dept] += 1
        s._doc_id = f"{s.topic.dept}-{dept_counter[s.topic.dept]:04d}"  # type: ignore[attr-defined]

    id_by_key: dict[tuple[str, str, str], str] = {}          # (topic, lang, office) -> HO doc, any type
    id_by_key_typed: dict[tuple[str, str, str, str], str] = {}  # + doc type
    for s in specs:
        id_by_key[(s.topic.key, s.lang, s.office_key)] = s._doc_id  # type: ignore[attr-defined]
        id_by_key_typed[(s.topic.key, s.lang, s.office_key, s.doc_type)] = s._doc_id  # type: ignore[attr-defined]

    for s in specs:
        # Seeded by topic+office ONLY - not doc_type, not language - so every
        # document describing the same real-world policy (Policy, Procedure,
        # FAQ, Quick Reference, English or Arabic) states the same facts,
        # instead of each document type or translation independently rolling
        # its own numbers. (Found the hard way: the English and Arabic
        # Policy for the same topic used to disagree - e.g. one said pages
        # are archived after 6 months, the other after 12.)
        facts_rng = random.Random(f"{SEED}:{s.topic.key}:{s.office_key}:facts")
        values = pick_params(s.topic, facts_rng, s.shift)

        rng2 = random.Random(f"{SEED}:{s.topic.key}:{s.doc_type}:{s.lang}:{s.office_key}")
        # ~12% of documents are an older, superseded version
        superseded = rng2.random() < 0.12
        version = f"{s.version_major}.{rng2.randint(0, 4)}"
        eff = base_date + timedelta(days=rng2.randint(0, 900))
        status = "Active"
        superseded_by = None
        if superseded:
            status = "Superseded"
            version = f"{max(1, s.version_major - 1)}.{rng2.randint(0, 6)}"
            eff = base_date + timedelta(days=rng2.randint(0, 400))
            # points to the current Head-Office document of the same topic and type
            tgt = id_by_key_typed.get((s.topic.key, s.lang, "HO", s.doc_type))
            if tgt and tgt != s._doc_id:  # type: ignore[attr-defined]
                superseded_by = tgt
            else:
                status, version = "Active", f"{s.version_major}.{rng2.randint(0, 4)}"

        s.superseded_by = superseded_by

        related_ids = []
        for rk in s.topic.related[:3]:
            rid = id_by_key.get((rk, s.lang, "HO"))
            if rid:
                related_ids.append(rid)

        body = make_body(s, values, s._doc_id, eff, version, status, related_ids)  # type: ignore[attr-defined]

        fname = f"{s._doc_id}_{slugify(s.topic.title_en)}_{s.doc_type.replace(' ', '')}.txt"  # type: ignore[attr-defined]
        path = out_dir / s.lang / fname
        path.write_text(body, encoding="utf-8")

        title = doc_title(s)

        rows.append({
            "doc_id": s._doc_id,  # type: ignore[attr-defined]
            "file": f"{s.lang}/{fname}",
            "title": title,
            "topic_key": s.topic.key,
            "department": DEPARTMENTS[s.topic.dept],
            "department_code": s.topic.dept,
            "type": s.doc_type,
            "language": s.lang,
            "office": s.office_en if s.lang == "en" else s.office_ar,
            "office_code": s.office_key,
            "version": version,
            "status": status,
            "effective_date": eff.isoformat(),
            "superseded_by": superseded_by or "",
            "related": " ".join(related_ids),
        })

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # small summary
    by_lang: dict[str, int] = {}
    by_dept: dict[str, int] = {}
    by_type: dict[str, int] = {}
    n_superseded = 0
    for r in rows:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        by_dept[r["department_code"]] = by_dept.get(r["department_code"], 0) + 1
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        n_superseded += r["status"] == "Superseded"

    print(f"Wrote {len(rows)} documents to {out_dir}")
    print(f"  manifest : {manifest}")
    print(f"  language : {dict(sorted(by_lang.items()))}")
    print(f"  type     : {dict(sorted(by_type.items()))}")
    print(f"  dept     : {dict(sorted(by_dept.items()))}")
    print(f"  superseded (old versions): {n_superseded}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the synthetic knowledge base.")
    ap.add_argument("--count", type=int, default=2000, help="number of documents (default 2000)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()
    build(args.out, args.count)


if __name__ == "__main__":
    main()
