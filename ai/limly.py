import joblib
import os
from typing import Any

class LimlyModel:
    _cache = {}

    @staticmethod
    def load(name: str, path: str) -> Any:
        if name not in LimlyModel._cache:
            if os.path.exists(path):
                LimlyModel._cache[name] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Limly model {name} not found at {path}")
        return LimlyModel._cache[name]

    @staticmethod
    def save(name: str, model: Any, path: str) -> None:
        joblib.dump(model, path)
        LimlyModel._cache[name] = model