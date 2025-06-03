# app/api/endpoints/reference_data.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from typing import List, Dict, Any, Optional
from mysql.connector import MySQLConnection
import io

from app.core.database import get_db_connection
from app.services.reference_data_service import ReferenceDataService

# API 라우터 생성
router = APIRouter()

@router.post("/upload")
async def upload_reference_data(
    file: UploadFile = File(...),
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    CSV 파일을 업로드하여 참조 데이터를 추가하는 엔드포인트
    
    Args:
        file: 업로드할 CSV 파일
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 처리 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")
    
    # 파일 내용 읽기
    contents = await file.read()
    csv_content = contents.decode('utf-8')
    
    service = ReferenceDataService(db)
    result = service.import_from_csv(csv_content)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    db.close()
    return result

@router.get("/")
def get_reference_data(
    age_group: Optional[str] = None,
    occupation: Optional[str] = None,
    region: Optional[str] = None,
    income_level: Optional[str] = None,
    gender: Optional[str] = None,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    참조 데이터를 조회하는 엔드포인트
    
    Args:
        age_group: 연령대 필터 (선택)
        occupation: 직업 필터 (선택)
        region: 지역 필터 (선택)
        income_level: 소득 수준 필터 (선택)
        gender: 성별 필터 (선택)
        db: 데이터베이스 연결 객체
    
    Returns:
        List[Dict[str, Any]]: 참조 데이터 목록
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = ReferenceDataService(db)
    result = service.get_reference_data(age_group, occupation, region, income_level, gender)
    
    db.close()
    return result

@router.post("/create-table")
def create_reference_data_table(db: MySQLConnection = Depends(get_db_connection)):
    """
    참조 데이터 테이블을 생성하는 엔드포인트 (데이터베이스 초기 설정용)
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = ReferenceDataService(db)
    service.create_tables()
    
    db.close()
    return {"message": "참조 데이터 테이블이 성공적으로 생성되었습니다."}