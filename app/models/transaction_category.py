# 파일 위치: app/models/transaction_category.py
# 설명: 거래 카테고리 정보를 표현하는 모델 클래스

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TransactionCategory:
    """
    거래 카테고리 정보를 표현하는 모델 클래스
    
    Attributes:
        id: 카테고리 고유 식별자
        name: 카테고리 이름
        description: 카테고리 설명
        created_at: 카테고리 생성 시간
    """
    id: Optional[int]
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None