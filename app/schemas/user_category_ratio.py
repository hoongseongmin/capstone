# app/schemas/user_category_ratio.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class UserCategoryRatioBase(BaseModel):
    """
    사용자별 카테고리 지출 비율 기본 정보를 위한 Pydantic 스키마
    
    Attributes:
        user_id: 사용자 ID
        category_id: 카테고리 ID
        ratio: 지출 비율
        avg_amount: 평균 지출 금액
        period_start: 기간 시작일
        period_end: 기간 종료일
    """
    user_id: int = Field(..., description="사용자 ID")
    category_id: int = Field(..., description="카테고리 ID")
    ratio: float = Field(..., description="지출 비율 (0.0 ~ 1.0)", ge=0, le=1)
    avg_amount: float = Field(..., description="평균 지출 금액", ge=0)
    period_start: datetime = Field(..., description="기간 시작일")
    period_end: datetime = Field(..., description="기간 종료일")

class UserCategoryRatioCreate(UserCategoryRatioBase):
    """
    사용자별 카테고리 지출 비율 생성을 위한 Pydantic 스키마
    """
    pass

class UserCategoryRatioResponse(UserCategoryRatioBase):
    """
    사용자별 카테고리 지출 비율 응답을 위한 Pydantic 스키마
    
    Attributes:
        id: 사용자별 카테고리 지출 비율 고유 식별자
        category_name: 카테고리 이름
        created_at: 생성 시간
        updated_at: 업데이트 시간
    """
    id: int
    category_name: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True