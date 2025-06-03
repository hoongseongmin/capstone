# app/api/endpoints/user_ratio.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from mysql.connector import MySQLConnection

from app.core.database import get_db_connection
from app.repositories.user_category_ratio_repository import UserCategoryRatioRepository

# API 라우터 생성
router = APIRouter()

@router.get("/user/{user_id}")
def get_user_category_ratios(
    user_id: int,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 카테고리별 지출 비율을 조회하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        period_start: 기간 시작일 (선택적)
        period_end: 기간 종료일 (선택적)
        db: 데이터베이스 연결 객체
    
    Returns:
        List[Dict[str, Any]]: 카테고리별 지출 비율 목록
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = UserCategoryRatioRepository(db)
    result = repository.get_by_user_id(user_id, period_start, period_end)
    
    db.close()
    return result

@router.post("/create-table")
def create_user_category_ratio_table(db: MySQLConnection = Depends(get_db_connection)):
    """
    사용자 카테고리 비율 테이블을 생성하는 엔드포인트 (데이터베이스 초기 설정용)
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = UserCategoryRatioRepository(db)
    repository.create_table()
    
    db.close()
    return {"message": "사용자 카테고리 비율 테이블이 성공적으로 생성되었습니다."}