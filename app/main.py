# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="동일 집단 소비습관 분석 API",
    description="사용자 데이터와 거래 데이터를 관리하고 동일 집단 소비 패턴을 분석하는 RESTful API",
    version="0.1.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함 주석 처리
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    """
    루트 경로에 대한 간단한 환영 메시지
    """
    return {
        "message": "동일 집단 소비습관 분석 시스템에 오신 것을 환영합니다!",
        "api_docs": "/docs",
        "api_redoc": "/redoc"
    }