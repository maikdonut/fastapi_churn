from fastapi import HTTPException


class ServiceException(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: str | None = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.error_details = details


class ModelNotTrainedException(ServiceException):
    def __init__(self, details: str | None = None):
        super().__init__(
            status_code=503,
            code="MODEL_NOT_TRAINED",
            message="Модель не обучена. Вызовите POST /model/train.",
            details=details,
        )


class DatasetNotFoundException(ServiceException):
    def __init__(self, details: str | None = None):
        super().__init__(
            status_code=503,
            code="DATASET_NOT_FOUND",
            message="Датасет не найден.",
            details=details,
        )


class DatasetEmptyException(ServiceException):
    def __init__(self):
        super().__init__(
            status_code=503,
            code="DATASET_EMPTY",
            message="Датасет пустой.",
        )


class TrainingFailedException(ServiceException):
    def __init__(self, details: str | None = None):
        super().__init__(
            status_code=500,
            code="TRAINING_FAILED",
            message="Ошибка обучения модели.",
            details=details,
        )


class PredictionFailedException(ServiceException):
    def __init__(self, details: str | None = None):
        super().__init__(
            status_code=500,
            code="PREDICTION_FAILED",
            message="Ошибка предсказания.",
            details=details,
        )