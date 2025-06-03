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

# 분류 관련 라우터
try:
    from app.api.endpoints import classification
    router.include_router(classification.router, prefix="/classification", tags=["classification"])
    print("✅ 분류 서비스 라우터가 성공적으로 등록되었습니다.")
except ImportError as e:
    print(f"⚠️ 분류 서비스 라우터 등록 실패: {e}")
    print("🔧 분류 서비스는 나중에 추가될 예정입니다.")
except Exception as e:
    print(f"❌ 분류 서비스 라우터 로딩 중 오류: {e}")

# 향후 추가될 라우터들을 위한 주석
# 참조 데이터 관련 라우터 (향후 추가 예정)
# from app.api.endpoints import reference_data
# router.include_router(reference_data.router, prefix="/reference-data", tags=["reference-data"])

# 사용자 카테고리 비율 관련 라우터 (향후 추가 예정)  
# from app.api.endpoints import user_ratio
# router.include_router(user_ratio.router, prefix="/user-ratios", tags=["user-ratios"])