# 파일 위치: app/repositories/transaction_category_repository.py
# 설명: 거래 카테고리에 대한 데이터베이스 접근 계층 클래스

from typing import List, Optional, Dict, Any
from mysql.connector import MySQLConnection
from app.models.transaction_category import TransactionCategory

class TransactionCategoryRepository:
    """
    거래 카테고리 데이터에 접근하기 위한 데이터 접근 계층
    
    이 클래스는 거래 카테고리 정보에 대한 CRUD 작업을 수행합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        TransactionCategoryRepository 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
    
    def create(self, category: TransactionCategory) -> int:
        """
        새로운 카테고리를 데이터베이스에 추가
        
        Args:
            category: 생성할 카테고리 객체
        
        Returns:
            int: 생성된 카테고리의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO transaction_categories (name, description) VALUES (%s, %s)",
            (category.name, category.description)
        )
        self.connection.commit()
        category_id = cursor.lastrowid
        cursor.close()
        return category_id
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        모든 카테고리 목록을 가져옴
        
        Returns:
            List[Dict[str, Any]]: 카테고리 정보 딕셔너리 리스트
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM transaction_categories ORDER BY name")
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_by_id(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        지정된 ID의 카테고리를 가져옴
        
        Args:
            category_id: 찾을 카테고리 ID
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 카테고리 정보 딕셔너리, 찾지 못한 경우 None
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM transaction_categories WHERE id = %s", (category_id,))
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        지정된 이름의 카테고리를 가져옴
        
        Args:
            name: 찾을 카테고리 이름
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 카테고리 정보 딕셔너리, 찾지 못한 경우 None
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM transaction_categories WHERE name = %s", (name,))
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def update(self, category_id: int, name: str, description: Optional[str] = None) -> bool:
        """
        지정된 ID의 카테고리 정보를 업데이트
        
        Args:
            category_id: 업데이트할 카테고리 ID
            name: 새 카테고리 이름
            description: 새 카테고리 설명
        
        Returns:
            bool: 업데이트 성공 여부
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE transaction_categories SET name = %s, description = %s WHERE id = %s",
            (name, description, category_id)
        )
        self.connection.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        return updated
    
    def delete(self, category_id: int) -> bool:
        """
        지정된 ID의 카테고리를 삭제
        
        Args:
            category_id: 삭제할 카테고리 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM transaction_categories WHERE id = %s", (category_id,))
        self.connection.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    
    def create_table(self) -> None:
        """
        카테고리 테이블을 생성 (존재하지 않는 경우)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transaction_categories (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.connection.commit()
        cursor.close()
    
    def initialize_default_categories(self) -> None:
        """
        기본 카테고리 초기화 (아직 없는 경우에만 추가)
        """
        default_categories = [
            ("식비", "음식, 식당, 카페, 간식 등"),
            ("교통비", "대중교통, 택시, 자동차 유지비 등"),
            ("주거비", "임대료, 관리비, 수도광열비 등"),
            ("통신비", "휴대폰, 인터넷, 케이블TV 등"),
            ("여가/문화", "영화, 공연, 취미활동 등"),
            ("쇼핑", "의류, 생활용품, 전자제품 등"),
            ("의료/건강", "병원비, 약국, 운동 등"),
            ("교육", "학비, 강의, 책, 학원 등"),
            ("여행", "숙박, 항공, 관광 등"),
            ("기타", "기타 분류되지 않은 지출")
        ]
        
        cursor = self.connection.cursor()
        
        for name, description in default_categories:
            # 이미 존재하는지 확인
            cursor.execute("SELECT COUNT(*) FROM transaction_categories WHERE name = %s", (name,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                cursor.execute(
                    "INSERT INTO transaction_categories (name, description) VALUES (%s, %s)",
                    (name, description)
                )
        
        self.connection.commit()
        cursor.close()