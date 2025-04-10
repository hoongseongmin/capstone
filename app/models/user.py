# app/models/user.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class User:
    """
    사용자 정보를 표현하는 모델 클래스
    
    Attributes:
        id: 사용자 고유 식별자
        username: 사용자 로그인 ID
        password_hash: 암호화된 비밀번호
        name: 사용자 이름
        age: 사용자 나이
        occupation: 사용자 직업
        address: 사용자 거주 지역
        gender: 사용자 성별
        contact: 사용자 연락처
        income_level: 사용자 소득 수준
        created_at: 사용자 정보 생성 시간
    """
    id: Optional[int]
    username: str  # 사용자 ID(key)
    password_hash: str  # 암호화된 비밀번호
    name: str  # 이름
    age: int  # 나이
    occupation: str  # 직업
    address: str  # 거주 지역
    gender: str  # 성별
    contact: str  # 연락처
    income_level: Optional[str] = None  # 소득 수준
    created_at: Optional[datetime] = None  # 생성 시간