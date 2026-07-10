import json
from app.core.paths import HISTORY_PATH


class TrainingHistoryService:
    def __init__(self):
        self._records: list[dict] = []

    def save(self) -> None:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def add(self, record: dict) -> None:
        self._records.append(record)
        self.save()

    def load(self) -> bool:
        try:
            with open(HISTORY_PATH, "r") as file:
                self._records = json.load(file)
            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False

    def get_all(self, model_type: str | None = None) -> list[dict]:
        if model_type is None:
            return self._records
        return [r for r in self._records if r["model_type"] == model_type]

history_service = TrainingHistoryService()