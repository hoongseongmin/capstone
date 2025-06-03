# app/models/user_category_ratio.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class UserCategoryRatio:
    """
    사용자별 카테고리 지출 비율을 표현하는 모델 클래스
    
    Attributes:
        id: 고유 식별자
        user_id: 사용자 ID
        category_id: 카테고리 ID
        ratio: 지출 비율 (0.0 ~ 1.0)
        avg_amount: 평균 지출 금액
        period_start: 기간 시작일
        period_end: 기간 종료일
        created_at: 생성 시간
        updated_at: 업데이트 시간
    """
    id: Optional[int]
    user_id: int
    category_id: int
    ratio: float
    avg_amount: float
    period_start: datetime
    period_end: datetime
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None