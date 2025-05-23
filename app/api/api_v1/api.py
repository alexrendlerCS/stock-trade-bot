from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Trade, Position, Performance
from typing import List
import logging

logger = logging.getLogger(__name__)

api_router = APIRouter()

@api_router.get("/trades/", response_model=List[dict])
def get_trades(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of trades"""
    trades = db.query(Trade).offset(skip).limit(limit).all()
    return trades

@api_router.get("/positions/", response_model=List[dict])
def get_positions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of current positions"""
    positions = db.query(Position).filter(Position.is_active == True).offset(skip).limit(limit).all()
    return positions

@api_router.get("/performance/", response_model=List[dict])
def get_performance(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get performance metrics"""
    performance = db.query(Performance).offset(skip).limit(limit).all()
    return performance

@api_router.get("/health/")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 