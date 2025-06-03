# app/models/reference_data.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ReferenceData:
    """
    동일 집단 참조 데이터를 표현하는 모델 클래스
    
    Attributes:
        id: 참조 데이터 고유 식별자
        age_group: 나이 그룹 (20대, 30대 등)
        occupation: 직업 분류
        region: 지역 분류
        income_level: 소득 수준 분류
        gender: 성별
        category_id: 카테고리 ID
        spending_ratio: 지출 비율 (0.0 ~ 1.0)
        avg_amount: 평균 지출 금액
        created_at: 데이터 생성 시간
        updated_at: 데이터 업데이트 시간
    """
    id: Optional[int]
    age_group: str
    occupation: str
    region: str
    income_level: str
    gender: str
    category_id: int
    spending_ratio: float
    avg_amount: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None