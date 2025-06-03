# app/api/endpoints/transactions.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from typing import List, Optional, Dict, Any
from datetime import datetime
from mysql.connector import MySQLConnection
from app.core.database import get_db_connection
from app.models.transaction import Transaction
from app.schemas.transaction import TransactionCreate, TransactionResponse, TransactionList
from app.services.transaction_service import TransactionService
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Form
import tempfile
import os
from app.services.classification_service import classification_service


# API 라우터 생성
router = APIRouter()

@router.post("/", response_model=dict)
def create_transaction(transaction: TransactionCreate, db: MySQLConnection = Depends(get_db_connection)):
    """
    새로운 거래를 생성하는 엔드포인트
    
    Args:
        transaction: 생성할 거래 정보
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 생성된 거래 ID와 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    transaction_model = Transaction(id=None, **transaction.dict())
    transaction_id = service.create_transaction(transaction_model)
    
    db.close()
    return {"id": transaction_id, "message": "거래 정보가 성공적으로 등록되었습니다."}

@router.get("/user/{user_id}", response_model=List[TransactionResponse])
def get_user_transactions(user_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    특정 사용자의 모든 거래 내역을 가져오는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        List[TransactionResponse]: 거래 객체 리스트
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    transactions = service.get_user_transactions(user_id)
    
    db.close()
    return transactions

@router.get("/{transaction_id}", response_model=TransactionResponse)
def get_transaction(transaction_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 거래를 가져오는 엔드포인트
    
    Args:
        transaction_id: 찾을 거래 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        TransactionResponse: 거래 객체
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 거래를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    transaction = service.get_transaction_by_id(transaction_id)
    
    if not transaction:
        db.close()
        raise HTTPException(status_code=404, detail=f"거래 ID {transaction_id}를 찾을 수 없습니다")
    
    db.close()
    return transaction

@router.delete("/{transaction_id}")
def delete_transaction(transaction_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 거래를 삭제하는 엔드포인트
    
    Args:
        transaction_id: 삭제할 거래 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 거래를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    transaction = service.get_transaction_by_id(transaction_id)
    
    if not transaction:
        db.close()
        raise HTTPException(status_code=404, detail=f"거래 ID {transaction_id}를 찾을 수 없습니다")
    
    service.delete_transaction(transaction_id)
    db.close()
    return {"message": f"거래 ID {transaction_id}가 성공적으로 삭제되었습니다"}

@router.get("/summary/{user_id}")
def get_transaction_summary(
    user_id: int, 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None,
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    특정 사용자의 카테고리별 거래 요약 정보를 가져오는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        start_date: 시작 날짜 (선택적)
        end_date: 종료 날짜 (선택적)
        db: 데이터베이스 연결 객체
    
    Returns:
        List[Dict[str, Any]]: 카테고리별 거래 요약 정보
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    summary = service.get_category_summary(user_id, start_date, end_date)
    
    db.close()
    return summary

@router.post("/ai-analyzed")
def process_ai_analyzed_transactions(
    user_id: int = Body(..., embed=True),
    transactions_data: List[Dict[str, Any]] = Body(..., embed=True),
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    AI 담당자로부터 전달받은 분석된 거래 데이터를 처리하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        transactions_data: AI 분석 결과 거래 데이터 리스트
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 처리 결과
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    result = service.process_ai_analyzed_transactions(user_id, transactions_data)
    
    db.close()
    return result

@router.post("/create-tables")
def create_transaction_tables(db: MySQLConnection = Depends(get_db_connection)):
    """
    거래 관련 테이블을 생성하는 엔드포인트 (데이터베이스 초기 설정용)
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = TransactionService(db)
    service.create_tables()
    
    db.close()
    return {"message": "거래 관련 테이블이 성공적으로 생성되었습니다."}

@router.post("/upload-nh-excel", response_model=dict)
async def upload_nh_excel(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    NH농협 거래내역 엑셀 파일을 업로드하고 자동 분류하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        file: 업로드할 NH 거래내역 엑셀 파일
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 처리 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    # 파일 형식 검증
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="엑셀 파일(.xlsx, .xls)만 업로드 가능합니다.")
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # AI 모델 초기화 (처음 사용시에만)
        classification_service.initialize_model()
        
        # NH 거래내역 파싱 및 분류
        transactions_data = classification_service.parse_nh_excel(temp_file_path)
        
        # 서비스 레이어를 통해 데이터베이스에 저장
        service = TransactionService(db)
        result = service.process_ai_analyzed_transactions(user_id, transactions_data)
        
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        
        db.close()
        
        return {
            "message": "NH 거래내역이 성공적으로 업로드되고 분류되었습니다.",
            "file_name": file.filename,
            "total_transactions": len(transactions_data),
            "processing_result": result
        }
        
    except Exception as e:
        # 임시 파일 정리
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        db.close()
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")