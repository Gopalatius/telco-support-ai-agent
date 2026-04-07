from fastapi import APIRouter, HTTPException

from telco_agent.api.dependencies import get_settings, get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    try:
        get_settings()
        is_healthy = get_vector_store().healthcheck()
    except Exception as error:
        raise HTTPException(
            status_code=503, detail="dependency healthcheck failed"
        ) from error

    if not is_healthy:
        raise HTTPException(status_code=503, detail="dependency healthcheck failed")

    return {"status": "ok"}
