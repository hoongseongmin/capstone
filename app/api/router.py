# app/api/router.py
from fastapi import APIRouter

# 메인 API 라우터 생성
router = APIRouter()

# 엔드포인트 임포트
from app.api.endpoints import users, transactions, categories, analysis

# 사용자 관련 라우터를 /users 경로로 포함
router.include_router(users.router, prefix="/users", tags=["users"])

# 거래 관련 라우터를 /transactions 경로로 포함
router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])

# 카테고리 관련 라우터를 /categories 경로로 포함
router.include_router(categories.router, prefix="/categories", tags=["categories"])

# 분석 관련 라우터를 /analysis 경로로 포함
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])