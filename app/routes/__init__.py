from .users import router as users_router
from .chat import router as chat_router
from .collections import router as collections_router
from .quizzes import router as quizzes_router
from .scrape import router as scrape_router

__all__ = ["users_router", "chat_router", "collections_router", "quizzes_router", "scrape_router"]
