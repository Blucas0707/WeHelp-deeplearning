from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.config import get_settings
from web.routers.index import router as index_router
from web.routers.predict import router as predict_router
from web.routers.feedback import router as feedback_router

app = FastAPI()
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(index_router)
app.include_router(predict_router)
app.include_router(feedback_router)
