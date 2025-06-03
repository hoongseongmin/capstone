# 파일 위치: app/api/endpoints/analysis.py
# 설명: 분석 서비스를 API 엔드포인트로 노출시키는 코드

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
from mysql.connector import MySQLConnection

from app.core.database import get_db_connection
from app.services.analysis_service import AnalysisService

router = APIRouter()

@router.get("/user/{user_id}/spending-pattern")
def analyze_spending_pattern(
    user_id: int, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 소비 패턴을 분석하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        start_date: 시작 날짜 (선택적)
        end_date: 종료 날짜 (선택적)
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 소비 패턴 분석 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.analyze_user_spending_pattern(user_id, start_date, end_date)
    
    db.close()
    return result

@router.get("/user/{user_id}/group-comparison/{criteria}")
def compare_with_group(
    user_id: int, 
    criteria: str,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    동일 그룹 내 다른 사용자들과 소비 패턴을 비교하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        criteria: 비교 기준 (나이, 직업, 지역, 소득 등)
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 그룹 비교 분석 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    valid_criteria = ['age', 'job', 'region', 'income', 'gender']
    if criteria not in valid_criteria:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 비교 기준입니다. 유효한 기준: {', '.join(valid_criteria)}")
    
    service = AnalysisService(db)
    result = service.compare_with_group(user_id, criteria)
    
    db.close()
    return result

@router.get("/user/{user_id}/monthly-trend")
def analyze_monthly_trend(
    user_id: int,
    months: int = 6,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 월별 소비 추세를 분석하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        months: 조회할 월 수 (기본값: 6)
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 월별 소비 추세 분석 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.analyze_monthly_trend(user_id, months)
    
    db.close()
    return result

@router.get("/user/{user_id}/anomalies")
def detect_anomalies(
    user_id: int,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 이상 소비 패턴을 감지하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 이상 소비 감지 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.detect_anomalies(user_id)
    
    db.close()
    return result

@router.get("/user/{user_id}/insights")
def generate_insights(
    user_id: int,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 소비 패턴에 대한 인사이트를 생성하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 소비 인사이트
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.generate_insights(user_id)
    
    db.close()
    return result

# 기존 파일에 추가할 코드

@router.post("/user/{user_id}/calculate-ratios")
def calculate_user_category_ratios(
    user_id: int,
    start_date: datetime,
    end_date: datetime, 
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 기간 동안의 사용자 카테고리별 지출 비율을 계산하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        start_date: 시작 날짜
        end_date: 종료 날짜
        db: 데이터베이스 연결 객체
    
    Returns:
        List[Dict[str, Any]]: 카테고리별 지출 비율
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.calculate_user_category_ratios(user_id, start_date, end_date)
    
    db.close()
    return result

@router.get("/user/{user_id}/reference-comparison")
def compare_with_reference_data(
    user_id: int,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    사용자의 지출 패턴과 참조 데이터를 비교하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 비교 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = AnalysisService(db)
    result = service.compare_with_reference_data(user_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    db.close()
    return result