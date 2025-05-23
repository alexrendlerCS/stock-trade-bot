from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.api_v1.api import api_router
from app.core.scheduler import scheduler

app = FastAPI(
    title="AI Stock Trading Bot",
    description="Autonomous stock trading bot with multiple strategies",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Start the scheduler
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Shutdown the scheduler
    scheduler.shutdown()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Stock Trading Bot API",
        "status": "operational",
        "version": "1.0.0"
    } 