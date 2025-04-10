# 파일 위치: app/schemas/transaction_category.py
# 설명: 거래 카테고리 관련 Pydantic 스키마 모델

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class CategoryBase(BaseModel):
    """
    카테고리 기본 정보를 위한 Pydantic 스키마
    
    Attributes:
        name: 카테고리 이름
        description: 카테고리 설명
    """
    name: str = Field(..., description="카테고리 이름")
    description: Optional[str] = Field(None, description="카테고리 설명")

class CategoryCreate(CategoryBase):
    """
    카테고리 생성을 위한 Pydantic 스키마
    """
    pass

class CategoryResponse(CategoryBase):
    """
    카테고리 정보 응답을 위한 Pydantic 스키마
    
    Attributes:
        id: 카테고리 고유 식별자
        created_at: 생성 시간
    """
    id: int
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True  # Pydantic v2에서는 orm_mode 대신 from_attributes 사용

class CategoryList(BaseModel):
    """
    카테고리 목록 응답을 위한 Pydantic 스키마
    
    Attributes:
        categories: 카테고리 객체 리스트
    """
    categories: List[CategoryResponse]