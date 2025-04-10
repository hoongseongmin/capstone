# app/repositories/user_repository.py
from typing import List, Optional, Dict, Any
from mysql.connector import MySQLConnection
from app.models.user import User

class UserRepository:
    """
    사용자 데이터에 접근하기 위한 데이터 접근 계층
    
    이 클래스는 사용자 정보에 대한 CRUD 작업을 수행합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        UserRepository 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
    
    def create(self, user: User) -> int:
        """
        새로운 사용자를 데이터베이스에 추가
        
        Args:
            user: 생성할 사용자 객체
        
        Returns:
            int: 생성된 사용자의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO users 
            (username, password_hash, name, age, occupation, address, gender, contact, income_level) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user.username, 
                user.password_hash, 
                user.name, 
                user.age, 
                user.occupation, 
                user.address, 
                user.gender, 
                user.contact,
                user.income_level
            )
        )
        self.connection.commit()
        user_id = cursor.lastrowid
        cursor.close()
        return user_id
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        모든 사용자 목록을 가져옴
        
        Returns:
            List[Dict[str, Any]]: 사용자 정보 딕셔너리 리스트
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        지정된 ID의 사용자를 가져옴
        
        Args:
            user_id: 찾을 사용자 ID
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 사용자 정보 딕셔너리, 찾지 못한 경우 None
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        지정된 사용자명의 사용자를 가져옴
        
        Args:
            username: 찾을 사용자명
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 사용자 정보 딕셔너리, 찾지 못한 경우 None
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def update(self, user_id: int, user_data: Dict[str, Any]) -> bool:
        """
        지정된 ID의 사용자 정보를 업데이트
        
        Args:
            user_id: 업데이트할 사용자 ID
            user_data: 업데이트할 데이터 딕셔너리
        
        Returns:
            bool: 업데이트 성공 여부
        """
        allowed_fields = {'name', 'age', 'occupation', 'address', 'gender', 'contact', 'income_level', 'password_hash'}
        update_fields = {k: v for k, v in user_data.items() if k in allowed_fields}
        
        if not update_fields:
            return False
        
        set_clause = ", ".join([f"{field} = %s" for field in update_fields.keys()])
        values = list(update_fields.values())
        values.append(user_id)
        
        cursor = self.connection.cursor()
        cursor.execute(
            f"UPDATE users SET {set_clause} WHERE id = %s",
            values
        )
        self.connection.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        return updated
    
    def delete(self, user_id: int) -> bool:
        """
        지정된 ID의 사용자를 삭제
        
        Args:
            user_id: 삭제할 사용자 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        self.connection.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    
    def create_table(self) -> None:
        """
        사용자 테이블을 생성 (존재하지 않는 경우)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            name VARCHAR(100) NOT NULL,
            age INT NOT NULL,
            occupation VARCHAR(100) NOT NULL,
            address VARCHAR(100) NOT NULL,
            gender VARCHAR(20) NOT NULL,
            contact VARCHAR(50) NOT NULL,
            income_level VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.connection.commit()
        cursor.close()