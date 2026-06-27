import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict
import os

MODEL_PATH = os.path.join("models", "multilingual-minilm")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

SEED_PHRASES: Dict[str, List[str]] = {
    "1.1": [
        "أزرع محاصيل وأبيع المنتجات الزراعية",
        "أربي الإبل والماعز وأبيع الحليب واللحوم",
        "أعمل في الزراعة والري والأرض",
        "أزرع التمر والخضروات والفواكه",
        "farming crops dates vegetables irrigation harvest",
        "I raise camels goats sheep cattle milk meat livestock",
        "agricultural land farming primary resources raw materials",
        "I grow food and sell it at the market",
        "I work with land and water irrigation farming",
    ],
    "1.2": [
        "أصنع منتجات يدوية في المنزل وأبيعها",
        "أطبخ وأحضر وجبات وحلويات وأبيعها",
        "أخيط وأصنع الملابس والأزياء التقليدية",
        "أصنع المجوهرات والحلي اليدوية",
        "I make handmade crafts jewellery accessories at home",
        "home cooking food production baking traditional food",
        "tailoring sewing embroidery handmade clothing",
        "I sell homemade products food or crafts",
        "I bake and make sweets and food from home",
    ],
    "1.3": [
        "أصنع منتجات في ورشتي وأبيعها",
        "أعمل في الحدادة والنجارة والتصنيع",
        "أملك ورشة وأريد توسيع إنتاجي",
        "small workshop manufacturing fabrication production",
        "furniture metalwork woodwork repair manufacturing",
        "I have a workshop and produce goods",
        "I manufacture small products in my facility",
    ],
    "2.1": [
        "أقدم خدمات تنظيف وصيانة للمنازل",
        "أصلح الأجهزة الكهربائية والسباكة",
        "أعمل في خدمات المنازل والمباني",
        "cleaning repair maintenance plumbing electrical home services",
        "I fix things and offer maintenance services to households",
        "I provide cleaning or repair services locally",
        "home repair handyman plumbing electrical",
    ],
    "2.2": [
        "أتنقل بين المناطق لتقديم الخدمات",
        "أقدم خدمات متنقلة وميدانية",
        "أعمل في التوصيل والنقل والشحن",
        "mobile repair on-site services traveling service provider field work",
        "I travel to customers to deliver my service",
        "delivery transport logistics mobile services",
        "I drive and deliver goods or services to people",
    ],
    "2.3": [
        "أعطي دروساً خصوصية وأدرّس الأطفال",
        "أهتم بالمسنين والأطفال ورعايتهم",
        "أقدم خدمات صحية ومجتمعية",
        "tutoring teaching childcare elderly care education community",
        "I teach or care for people in the community",
        "I give lessons or provide care for children elderly",
        "private tutor teacher community support",
    ],
    "2.4": [
        "أقدم خدمات استشارية ومحاسبية وإدارية",
        "أعمل في التصميم والتسويق والإعلام",
        "أقدم خدمات قانونية أو مالية",
        "consulting accounting design administrative professional services",
        "I offer expert advice or professional services to businesses",
        "I work in marketing design IT consulting",
        "professional freelance consulting digital services",
    ],
    "3.1": [
        "أريد فتح محل وبيع البضائع والمنتجات",
        "أوزع منتجات في المنطقة وأبيعها",
        "أريد فتح دكان أو متجر",
        "shop store retail sell goods market trading distribution",
        "I want to open a shop or distribute products locally",
        "I sell products in a shop or market stall",
        "retail store selling products grocery",
    ],
    "3.2": [
        "أنظم فعاليات ومناسبات وتجمعات",
        "أقدم تجارب ثقافية وتراثية",
        "أعمل في الفعاليات الاجتماعية والترفيه",
        "events cultural activities community gatherings experiences",
        "I organise community events or cultural experiences",
        "I run social events weddings ceremonies community activities",
        "cultural heritage events entertainment local",
    ],
    "3.3": [
        "أريد استقبال السياح وتقديم جولات في الصحراء",
        "أقدم تجارب تخييم ومشاهدة النجوم",
        "أريد العمل في السياحة والطبيعة",
        "desert tourism camping stargazing guided tours nature experiences",
        "I want to guide tourists in the desert or offer outdoor experiences",
        "I offer eco-tourism desert camping nature tours",
        "tourism guide outdoor adventure desert safari",
    ],
}

MACRO_GROUP_MAP = {
    "1.1": "1", "1.2": "1", "1.3": "1",
    "2.1": "2", "2.2": "2", "2.3": "2", "2.4": "2",
    "3.1": "3", "3.2": "3", "3.3": "3",
}

_model = None
_embeddings: Dict[str, np.ndarray] = {}


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME
        _model = SentenceTransformer(path)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH, exist_ok=True)
            _model.save(MODEL_PATH)
    return _model


def _get_embeddings() -> Dict[str, np.ndarray]:
    global _embeddings
    if not _embeddings:
        model = _load_model()
        for cat_id, phrases in SEED_PHRASES.items():
            vecs = model.encode(phrases, normalize_embeddings=True)
            _embeddings[cat_id] = np.mean(vecs, axis=0)
            norm = np.linalg.norm(_embeddings[cat_id])
            if norm > 0:
                _embeddings[cat_id] /= norm
    return _embeddings


def classify(text: str) -> Dict:
    """
    Returns:
      {
        "subcategory": "1.1",
        "macro_group": "1",
        "confidence": 0.87,
        "confidence_level": "high" | "medium" | "low",
        "top2": [("1.1", 0.87), ("1.2", 0.71)],
        "all_scores": {"1.1": 0.87, ...}
      }
    """
    model = _load_model()
    embeddings = _get_embeddings()

    query_vec = model.encode([text], normalize_embeddings=True)[0]

    scores: Dict[str, float] = {}
    for cat_id, cat_vec in embeddings.items():
        score = float(np.dot(query_vec, cat_vec))
        scores[cat_id] = max(0.0, score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = ranked[0]
    top2 = ranked[:2]

    if top_score >= 0.70:
        level = "high"
    elif top_score >= 0.45:
        level = "medium"
    else:
        level = "low"

    return {
        "subcategory": top_cat,
        "macro_group": MACRO_GROUP_MAP[top_cat],
        "confidence": round(top_score, 4),
        "confidence_level": level,
        "top2": top2,
        "all_scores": scores,
    }


def preload():
    """Call at startup to preload model and embeddings."""
    _load_model()
    _get_embeddings()