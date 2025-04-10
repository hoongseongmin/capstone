# 파일 위치: app/core/config.py
# 설명: 애플리케이션 설정 관리 모듈

from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    환경 변수 또는 .env 파일에서 설정을 로드합니다.
    """
    APP_NAME: str = "동일 집단 소비습관 분석 API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "사용자 데이터와 거래 데이터를 관리하고 동일 집단 소비 패턴을 분석하는 RESTful API"
    
    # 데이터베이스 설정
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "1234"
    DB_NAME: str = "finance_app"
    
    # CORS 설정
    CORS_ORIGINS: list = ["*"]
    
    # JWT 설정
    SECRET_KEY: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 설정 인스턴스 생성
settings = Settings()