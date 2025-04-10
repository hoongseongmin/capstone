# 파일 위치: app/models/transaction.py
# 설명: 거래 정보를 표현하는 모델 클래스

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Transaction:
    """
    거래 정보를 표현하는 모델 클래스
    
    Attributes:
        id: 거래 고유 식별자
        user_id: 사용자 ID
        category_id: 카테고리 ID
        amount: 거래 금액
        transaction_date: 거래 날짜
        description: 거래 설명
        payment_method: 결제 수단
        created_at: 생성 시간
    """
    id: Optional[int]
    user_id: int
    category_id: int
    amount: float
    transaction_date: datetime
    description: Optional[str] = None
    payment_method: Optional[str] = None
    created_at: Optional[datetime] = None