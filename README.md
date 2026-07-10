# Churn Prediction Service

REST API для обучения модели оттока клиентов (churn) и batch-предсказаний на основе FastAPI и scikit-learn.

Поддерживаемые модели: `logreg` (логистическая регрессия) и `random_forest` (случайный лес).

Интерактивная документация: [http://localhost:8000/docs](http://localhost:8000/docs)

## Структура проекта

```
assignment_fastapi/
├── app/
│   ├── main.py              # точка входа FastAPI
│   ├── api/routes/          # HTTP-эндпоинты
│   ├── core/                # пути к файлам и исключения
│   ├── schemas/             # Pydantic-модели
│   └── services/            # ML pipeline: dataset, preprocessing, model, history
├── data/
│   └── churn_dataset.csv    # обучающий датасет
├── models/                  # артефакты (модель, история обучений)
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Формат датасета

Файл `data/churn_dataset.csv` содержит 2000 строк. Train/test split 80/20 со стратификацией по целевой переменной.

| Колонка | Тип | Описание |
|---------|-----|----------|
| `monthly_fee` | float | Абонентская плата |
| `usage_hours` | float | Часы использования |
| `support_requests` | int | Обращения в поддержку |
| `account_age_months` | int | Возраст аккаунта (мес.) |
| `failed_payments` | int | Неудачные платежи |
| `region` | str | `europe`, `america`, `asia`, `africa` |
| `device_type` | str | `mobile`, `desktop`, `tablet` |
| `payment_method` | str | `card`, `paypal`, `crypto` |
| `autopay_enabled` | int | 0 или 1 |
| `churn` | int | Целевая переменная: 0 — остался, 1 — ушёл |

## Запуск локально

Требования: Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run uvicorn app.main:app --reload --port 8000
```

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
```

> Артефакты модели в контейнере эфемерны. После старта нужно обучить модель через `POST /model/train`.

## API-эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Статус сервиса |
| GET | `/health` | Проверка готовности (модель + датасет) |
| GET | `/dataset/preview?n=5` | Превью датасета |
| GET | `/dataset/info` | Статистика датасета |
| GET | `/dataset/split-info` | Информация о train/test split |
| POST | `/model/train` | Обучение модели |
| GET | `/model/status` | Статус и метрики модели |
| GET | `/model/schema` | Схема входных признаков |
| GET | `/model/metrics` | История обучений |
| POST | `/predict/` | Batch-предсказание |

## Примеры запросов

### Обучение модели (logreg)

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"max_iter": 200}}'
```

Ответ:

```json
{
  "accuracy": 0.8525,
  "f1": 0.7234,
  "roc_auc": 0.9102,
  "train_size": 1600,
  "test_size": 400
}
```

### Обучение модели (random forest)

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 100}}'
```

### Предсказание

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '[{
    "monthly_fee": 19.99,
    "usage_hours": 42.5,
    "support_requests": 2,
    "account_age_months": 14,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1
  }]'
```

Ответ:

```json
[
  {
    "churn": 0,
    "churn_probability": 0.23,
    "churn_label": "stay"
  }
]
```

## Тесты

```bash
uv run pytest
```
