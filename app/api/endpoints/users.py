# app/api/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any
from mysql.connector import MySQLConnection
from app.core.database import get_db_connection
from app.schemas.user import UserCreate, UserResponse, UserLogin
from app.services.user_service import UserService

# API 라우터 생성
router = APIRouter()

@router.post("/", response_model=Dict[str, Any])
def create_user(user: UserCreate, db: MySQLConnection = Depends(get_db_connection)):
    """
    새로운 사용자를 생성하는 엔드포인트
    
    Args:
        user: 생성할 사용자 정보
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 생성된 사용자 ID와 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 중복 사용자명 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    
    # 이미 존재하는 사용자명인지 확인
    existing_user = service.repository.get_by_username(user.username)
    if existing_user:
        db.close()
        raise HTTPException(status_code=400, detail=f"사용자명 '{user.username}'은 이미 사용 중입니다")
    
    user_id = service.create_user(
        username=user.username,
        password=user.password,
        name=user.name,
        age=user.age,
        occupation=user.occupation,
        address=user.address,
        gender=user.gender,
        contact=user.contact,
        income_level=user.income_level
    )
    
    db.close()
    return {"id": user_id, "message": "사용자 정보가 성공적으로 등록되었습니다."}

@router.post("/login", response_model=Dict[str, Any])
def login_user(user_login: UserLogin, db: MySQLConnection = Depends(get_db_connection)):
    """
    사용자 로그인 처리 엔드포인트
    
    Args:
        user_login: 로그인 정보
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 로그인 성공 시 사용자 정보와 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 인증 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    user = service.verify_user(user_login.username, user_login.password)
    
    if not user:
        db.close()
        raise HTTPException(status_code=401, detail="사용자명 또는 비밀번호가 올바르지 않습니다")
    
    db.close()
    return {"user": user, "message": "로그인이 성공적으로 완료되었습니다."}

@router.get("/", response_model=List[UserResponse])
def get_users(db: MySQLConnection = Depends(get_db_connection)):
    """
    모든 사용자 목록을 가져오는 엔드포인트
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        List[UserResponse]: 사용자 객체 리스트
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    users = service.get_all_users()
    
    db.close()
    return users

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 사용자를 가져오는 엔드포인트
    
    Args:
        user_id: 찾을 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        UserResponse: 사용자 객체
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 사용자를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    user = service.get_user_by_id(user_id)
    
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail=f"사용자 ID {user_id}를 찾을 수 없습니다")
    
    db.close()
    return user

@router.put("/{user_id}", response_model=Dict[str, Any])
def update_user(user_id: int, user_data: Dict[str, Any] = Body(...), db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 사용자 정보를 업데이트하는 엔드포인트
    
    Args:
        user_id: 업데이트할 사용자 ID
        user_data: 업데이트할 데이터
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 사용자를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    
    # 사용자가 존재하는지 확인
    user = service.get_user_by_id(user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail=f"사용자 ID {user_id}를 찾을 수 없습니다")
    
    # 사용자명 변경 시 중복 확인
    if 'username' in user_data and user_data['username'] != user['username']:
        existing_user = service.repository.get_by_username(user_data['username'])
        if existing_user:
            db.close()
            raise HTTPException(status_code=400, detail=f"사용자명 '{user_data['username']}'은 이미 사용 중입니다")
    
    success = service.update_user(user_id, user_data)
    
    db.close()
    return {"message": f"사용자 ID {user_id}가 성공적으로 업데이트되었습니다"}

@router.delete("/{user_id}")
def delete_user(user_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 사용자를 삭제하는 엔드포인트
    
    Args:
        user_id: 삭제할 사용자 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 사용자를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    
    # 사용자가 존재하는지 확인
    user = service.get_user_by_id(user_id)
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail=f"사용자 ID {user_id}를 찾을 수 없습니다")
    
    service.delete_user(user_id)
    
    db.close()
    return {"message": f"사용자 ID {user_id}가 성공적으로 삭제되었습니다"}

@router.post("/create-table/")
def create_users_table(db: MySQLConnection = Depends(get_db_connection)):
    """
    사용자 테이블을 생성하는 엔드포인트 (데이터베이스 초기 설정용)
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    service = UserService(db)
    service.create_users_table()
    
    db.close()
    return {"message": "사용자 테이블이 성공적으로 생성되었습니다."}