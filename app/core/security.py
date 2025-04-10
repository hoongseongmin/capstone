# 파일 위치: app/core/security.py
# 설명: 보안 관련 유틸리티 함수

import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from app.core.config import settings
import hashlib

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    JWT 액세스 토큰 생성
    
    Args:
        data: 토큰에 인코딩할 데이터
        expires_delta: 토큰 만료 시간 (선택적)
    
    Returns:
        str: 생성된 JWT 토큰
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    비밀번호 검증
    
    Args:
        plain_password: 평문 비밀번호
        hashed_password: 해시된 비밀번호
    
    Returns:
        bool: 비밀번호 일치 여부
    """
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_password_hash(password: str) -> str:
    """
    비밀번호 해싱
    
    Args:
        password: 평문 비밀번호
    
    Returns:
        str: 해시된 비밀번호
    """
    return hashlib.sha256(password.encode()).hexdigest()