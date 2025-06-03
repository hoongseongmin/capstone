# app/schemas/reference_data.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ReferenceDataBase(BaseModel):
    """
    참조 데이터 기본 정보를 위한 Pydantic 스키마
    
    Attributes:
        age_group: 나이 그룹
        occupation: 직업 분류
        region: 지역 분류
        income_level: 소득 수준
        gender: 성별
        category_id: 카테고리 ID
        spending_ratio: 지출 비율
        avg_amount: 평균 지출 금액
    """
    age_group: str = Field(..., description="나이 그룹 (20대, 30대 등)")
    occupation: str = Field(..., description="직업 분류")
    region: str = Field(..., description="지역 분류")
    income_level: str = Field(..., description="소득 수준 분류")
    gender: str = Field(..., description="성별")
    category_id: int = Field(..., description="카테고리 ID")
    spending_ratio: float = Field(..., description="지출 비율 (0.0 ~ 1.0)", ge=0, le=1)
    avg_amount: float = Field(..., description="평균 지출 금액", gt=0)

class ReferenceDataCreate(ReferenceDataBase):
    """
    참조 데이터 생성을 위한 Pydantic 스키마
    """
    pass

class ReferenceDataResponse(ReferenceDataBase):
    """
    참조 데이터 응답을 위한 Pydantic 스키마
    
    Attributes:
        id: 참조 데이터 고유 식별자
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