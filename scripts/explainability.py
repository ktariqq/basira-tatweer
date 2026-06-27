from typing import Dict


def _accessibility_label(val: float, lang: str) -> str:
    if val >= 0.70:
        return "جيدة" if lang == "ar" else "good"
    elif val >= 0.40:
        return "متوسطة" if lang == "ar" else "moderate"
    else:
        return "محدودة" if lang == "ar" else "limited"


def _competition_label(val: float, lang: str) -> str:
    # val is inverted penalty: high = low competition = good
    if val >= 0.70:
        return "منخفضة، وهو مؤشر إيجابي" if lang == "ar" else "low — a positive sign"
    elif val >= 0.40:
        return "متوسطة" if lang == "ar" else "moderate"
    else:
        return "مرتفعة" if lang == "ar" else "high"


def _demand_label(score: int, lang: str) -> str:
    if score >= 65:
        return "مرتفع" if lang == "ar" else "High"
    elif score >= 40:
        return "متوسط" if lang == "ar" else "Medium"
    else:
        return "منخفض" if lang == "ar" else "Low"


def generate_explanation(signals: Dict, category: str, lang: str = "ar") -> str:
    score = signals.get("demand_score", 0)
    agri = signals.get("agricultural_density", 0)
    pop = signals.get("population_proxy", 0)
    road = signals.get("road_accessibility", 0)
    comp = signals.get("competition_penalty", 0)
    rsrc = signals.get("resource_suitability", 0)

    demand_lbl = _demand_label(score, lang)
    acc_lbl = _accessibility_label(road, lang)
    comp_lbl = _competition_label(comp, lang)
    market_opp = "جيدة" if comp >= 0.60 else "متوسطة"
    market_opp_en = "good" if comp >= 0.60 else "moderate"

    if lang == "ar":
        return (
            f"حصل مشروعك على مؤشر طلب {score}/100 ({demand_lbl}). "
            f"المنطقة تحتوي على {int(agri * 100)}٪ أراضٍ ذات كثافة زراعية ضمن نطاق التحليل — "
            f"وهو مؤشر {'مرتفع' if agri > 0.3 else 'معتدل'} لهذا النشاط. "
            f"إمكانية الوصول إلى الطرق المعبّدة {acc_lbl}. "
            f"المنافسة في المنطقة {comp_lbl}، مما يعني فرصة سوقية {market_opp}. "
            f"ملاءمة الموارد الطبيعية للنشاط: {int(rsrc * 100)}٪."
        )
    else:
        return (
            f"Your business idea scored {score}/100 ({demand_lbl} demand). "
            f"The area shows {int(agri * 100)}% agricultural land density — "
            f"{'strong' if agri > 0.3 else 'moderate'} signal for this activity type. "
            f"Road accessibility is {acc_lbl}. "
            f"Competition in the area is {comp_lbl}, indicating a {market_opp_en} market opportunity. "
            f"Natural resource suitability for this category: {int(rsrc * 100)}%."
        )


def generate_full_explanation(signals: Dict, category: str) -> Dict[str, str]:
    return {
        "ar": generate_explanation(signals, category, lang="ar"),
        "en": generate_explanation(signals, category, lang="en"),
    }