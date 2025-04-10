# 파일 위치: app/schemas/transaction.py
# 설명: 거래 관련 Pydantic 스키마 모델

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TransactionBase(BaseModel):
    """
    거래 기본 정보를 위한 Pydantic 스키마
    
    Attributes:
        user_id: 사용자 ID
        category_id: 카테고리 ID
        amount: 거래 금액
        transaction_date: 거래 날짜
        payment_method: 결제 수단
        description: 거래 설명
    """
    user_id: int = Field(..., description="사용자 ID")
    category_id: int = Field(..., description="카테고리 ID")
    amount: float = Field(..., description="거래 금액", gt=0)
    transaction_date: datetime = Field(..., description="거래 날짜")
    payment_method: Optional[str] = Field(None, description="결제 수단")
    description: Optional[str] = Field(None, description="거래 설명")

class TransactionCreate(TransactionBase):
    """
    거래 생성을 위한 Pydantic 스키마
    """
    pass

class TransactionResponse(TransactionBase):
    """
    거래 정보 응답을 위한 Pydantic 스키마
    
    Attributes:
        id: 거래 고유 식별자
        created_at: 생성 시간
    """
    id: int
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True  # Pydantic v2에서는 orm_mode 대신 from_attributes 사용

class TransactionList(BaseModel):
    """
    거래 목록 응답을 위한 Pydantic 스키마
    
    Attributes:
        transactions: 거래 객체 리스트
    """
    transactions: List[TransactionResponse]