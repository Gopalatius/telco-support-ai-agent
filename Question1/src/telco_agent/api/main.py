from contextlib import asynccontextmanager

from fastapi import FastAPI

from telco_agent.api.routes.chat import router as chat_router
from telco_agent.api.routes.health import router as health_router
from telco_agent.infrastructure.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    yield


app = FastAPI(title="telco-customer-service-agent", lifespan=lifespan)
app.include_router(health_router)
app.include_router(chat_router)
