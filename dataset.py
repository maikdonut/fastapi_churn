from pathlib import Path
import pandas as pd
from schemas import DatasetRowChurn

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"

class ChurnDatasetService:
    """Загружает и хранит churn-датасет, отдаёт превью и статистику."""

    def __init__(self, csv_path: Path = DATASET_PATH):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Читает CSV в DataFrame и кеширует его в памяти."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Возвращает закешированный DataFrame, загружая его при первом обращении."""
        if self._df is None:
            self.load()
        return self._df

    def preview(self, n: int = 5) -> list[DatasetRowChurn]:
        """Возвращает первые n строк, провалидированных через DatasetRowChurn."""
        rows = self.df.head(n).to_dict(orient="records")
        return [DatasetRowChurn(**row) for row in rows]

    def validate_all(self) -> None:
        """Прогоняет весь датасет через DatasetRowChurn, чтобы проверить целостность данных."""
        records = self.df.to_dict(orient="records")
        for i, row in enumerate(records):
            try:
                DatasetRowChurn(**row)
            except Exception as e:
                raise ValueError(f"Row {i} failed validation: {e}") from e

    def info(self) -> dict:
        """Собирает базовую статистику по датасету: размеры, признаки, распределение churn."""
        df = self.df
        feature_columns = [col for col in df.columns if col != "churn"]

        return {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "feature_names": feature_columns,
            "churn_distribution": df["churn"].value_counts().to_dict(),
        }


dataset_service = ChurnDatasetService()
