# app/api/endpoints/categories.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from mysql.connector import MySQLConnection
from app.core.database import get_db_connection
from app.models.transaction_category import TransactionCategory
from app.schemas.transaction_category import CategoryCreate, CategoryResponse
from app.repositories.transaction_category_repository import TransactionCategoryRepository

# API 라우터 생성
router = APIRouter()

@router.get("/", response_model=List[CategoryResponse])
def get_categories(db: MySQLConnection = Depends(get_db_connection)):
    """
    모든 거래 카테고리 목록을 가져오는 엔드포인트
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        List[CategoryResponse]: 카테고리 객체 리스트
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    categories = repository.get_all()
    
    db.close()
    return categories

@router.post("/", response_model=dict)
def create_category(category: CategoryCreate, db: MySQLConnection = Depends(get_db_connection)):
    """
    새로운 거래 카테고리를 생성하는 엔드포인트
    
    Args:
        category: 생성할 카테고리 정보
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 생성된 카테고리 ID와 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    
    # 이미 존재하는 카테고리인지 확인
    existing = repository.get_by_name(category.name)
    if existing:
        db.close()
        raise HTTPException(status_code=400, detail=f"카테고리 '{category.name}'은(는) 이미 존재합니다")
    
    category_model = TransactionCategory(id=None, **category.dict())
    category_id = repository.create(category_model)
    
    db.close()
    return {"id": category_id, "message": "카테고리가 성공적으로 생성되었습니다."}

@router.get("/{category_id}", response_model=CategoryResponse)
def get_category(category_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 카테고리를 가져오는 엔드포인트
    
    Args:
        category_id: 찾을 카테고리 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        CategoryResponse: 카테고리 객체
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 카테고리를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    category = repository.get_by_id(category_id)
    
    if not category:
        db.close()
        raise HTTPException(status_code=404, detail=f"카테고리 ID {category_id}를 찾을 수 없습니다")
    
    db.close()
    return category

@router.put("/{category_id}", response_model=dict)
def update_category(category_id: int, category: CategoryCreate, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 카테고리를 업데이트하는 엔드포인트
    
    Args:
        category_id: 업데이트할 카테고리 ID
        category: 업데이트할 카테고리 정보
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 카테고리를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    
    # 카테고리가 존재하는지 확인
    existing = repository.get_by_id(category_id)
    if not existing:
        db.close()
        raise HTTPException(status_code=404, detail=f"카테고리 ID {category_id}를 찾을 수 없습니다")
    
    # 이름이 중복되는지 확인
    name_check = repository.get_by_name(category.name)
    if name_check and name_check['id'] != category_id:
        db.close()
        raise HTTPException(status_code=400, detail=f"카테고리 '{category.name}'은(는) 이미 존재합니다")
    
    repository.update(category_id, category.name, category.description)
    
    db.close()
    return {"message": f"카테고리 ID {category_id}가 성공적으로 업데이트되었습니다"}

@router.delete("/{category_id}")
def delete_category(category_id: int, db: MySQLConnection = Depends(get_db_connection)):
    """
    지정된 ID의 카테고리를 삭제하는 엔드포인트
    
    Args:
        category_id: 삭제할 카테고리 ID
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 또는 카테고리를 찾을 수 없을 때 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    
    # 카테고리가 존재하는지 확인
    existing = repository.get_by_id(category_id)
    if not existing:
        db.close()
        raise HTTPException(status_code=404, detail=f"카테고리 ID {category_id}를 찾을 수 없습니다")
    
    # 기본 카테고리('기타') 삭제 방지
    if existing['name'] == '기타':
        db.close()
        raise HTTPException(status_code=400, detail="'기타' 카테고리는 삭제할 수 없습니다")
    
    repository.delete(category_id)
    
    db.close()
    return {"message": f"카테고리 ID {category_id}가 성공적으로 삭제되었습니다"}

@router.post("/create-table")
def create_categories_table(db: MySQLConnection = Depends(get_db_connection)):
    """
    카테고리 테이블을 생성하는 엔드포인트 (데이터베이스 초기 설정용)
    
    Args:
        db: 데이터베이스 연결 객체
    
    Returns:
        dict: 성공 메시지
    
    Raises:
        HTTPException: 데이터베이스 연결 실패 시 발생
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    repository = TransactionCategoryRepository(db)
    repository.create_table()
    repository.initialize_default_categories()
    
    db.close()
    return {"message": "카테고리 테이블이 성공적으로 생성되고 초기화되었습니다."}