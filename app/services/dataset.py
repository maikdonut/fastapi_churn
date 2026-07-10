import logging
import pandas as pd

from app.core.paths import DATASET_PATH
from app.schemas.churn import DatasetRowChurn

logger = logging.getLogger(__name__)


class ChurnDatasetService:
    """Загружает и хранит churn-датасет, отдаёт превью и статистику."""

    def __init__(self, csv_path=DATASET_PATH):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Читает CSV в DataFrame и кеширует его в памяти."""
        if not self.csv_path.exists():
            logger.error(f"Dataset file not found: {self.csv_path}")
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self._df = df
        logger.info(f"Датасет загружен: {len(df)} строк, {len(df.columns)} столбцов из {self.csv_path}")
        return df

    @property
    def is_loaded(self) -> bool:
        return self._df is not None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    def preview(self, n: int = 5) -> list[DatasetRowChurn]:
        rows = self.df.head(n).to_dict(orient="records")
        return [DatasetRowChurn(**row) for row in rows]

    def info(self) -> dict:
        df = self.df
        feature_columns = [col for col in df.columns if col != "churn"]
        return {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "feature_names": feature_columns,
            "churn_distribution": df["churn"].value_counts().to_dict(),
        }


dataset_service = ChurnDatasetService()