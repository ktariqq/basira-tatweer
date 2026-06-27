import json
import os
from typing import Dict

LICENSE_PATH = os.path.join("config", "licenses.json")

_licenses: Dict = {}


def _load():
    global _licenses
    if not _licenses:
        with open(LICENSE_PATH, "r", encoding="utf-8") as f:
            _licenses = json.load(f)


def get_license(subcategory: str) -> Dict:
    _load()
    return _licenses.get(subcategory, _licenses.get("default", {}))