# app/schemas/user.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    """
    사용자 기본 정보를 위한 Pydantic 스키마
    
    Attributes:
        username: 사용자 로그인 ID
        name: 이름
        age: 나이
        occupation: 직업
        address: 거주 지역
        gender: 성별
        contact: 연락처
        income_level: 소득 수준
    """
    username: str = Field(..., description="사용자 로그인 ID")
    name: str = Field(..., description="사용자 이름")
    age: int = Field(..., description="사용자 나이", ge=1, le=120)
    occupation: str = Field(..., description="사용자 직업")
    address: str = Field(..., description="사용자 거주 지역")
    gender: str = Field(..., description="사용자 성별")
    contact: str = Field(..., description="사용자 연락처")
    income_level: Optional[str] = Field(None, description="사용자 소득 수준")

class UserCreate(UserBase):
    """
    사용자 생성을 위한 Pydantic 스키마
    
    Attributes:
        password: 비밀번호
    """
    password: str = Field(..., description="사용자 비밀번호", min_length=6)

class UserLogin(BaseModel):
    """
    사용자 로그인을 위한 Pydantic 스키마
    
    Attributes:
        username: 사용자 로그인 ID
        password: 비밀번호
    """
    username: str = Field(..., description="사용자 로그인 ID")
    password: str = Field(..., description="사용자 비밀번호")

class UserResponse(UserBase):
    """
    사용자 정보 응답을 위한 Pydantic 스키마
    
    Attributes:
        id: 사용자 고유 식별자
        created_at: 생성 시간
    """
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True    # ORM 객체를 Pydantic 모델로 변환 가능

class UserList(BaseModel):
    """
    사용자 목록 응답을 위한 Pydantic 스키마
    
    Attributes:
        users: 사용자 객체 리스트
    """
    users: List[UserResponse]