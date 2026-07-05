from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "ml churn service is running"}
