# app/repositories/reference_data_repository.py
from typing import List, Optional, Dict, Any, Tuple
from mysql.connector import MySQLConnection
from app.models.reference_data import ReferenceData

class ReferenceDataRepository:
    """
    참조 데이터에 접근하기 위한 데이터 접근 계층
    
    이 클래스는 참조 데이터에 대한 CRUD 작업을 수행합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        ReferenceDataRepository 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
    
    def create(self, data: ReferenceData) -> int:
        """
        새로운 참조 데이터를 데이터베이스에 추가
        
        Args:
            data: 생성할 참조 데이터 객체
        
        Returns:
            int: 생성된 데이터의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO reference_data 
            (age_group, occupation, region, income_level, gender, category_id, 
            spending_ratio, avg_amount) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                data.age_group,
                data.occupation,
                data.region,
                data.income_level,
                data.gender,
                data.category_id,
                data.spending_ratio,
                data.avg_amount
            )
        )
        self.connection.commit()
        data_id = cursor.lastrowid
        cursor.close()
        return data_id
    
    def create_many(self, data_list: List[ReferenceData]) -> int:
        """
        여러 참조 데이터를 한 번에 데이터베이스에 추가
        
        Args:
            data_list: 생성할 참조 데이터 객체 리스트
        
        Returns:
            int: 생성된 데이터의 수
        """
        cursor = self.connection.cursor()
        values = [
            (
                d.age_group,
                d.occupation,
                d.region,
                d.income_level,
                d.gender,
                d.category_id,
                d.spending_ratio,
                d.avg_amount
            ) 
            for d in data_list
        ]
        
        cursor.executemany(
            """
            INSERT INTO reference_data 
            (age_group, occupation, region, income_level, gender, category_id, 
            spending_ratio, avg_amount) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            values
        )
        self.connection.commit()
        count = cursor.rowcount
        cursor.close()
        return count
    
    def get_by_demographic(self, age_group: str, occupation: str, region: str, 
                         income_level: str, gender: str) -> List[Dict[str, Any]]:
        """
        인구통계학적 특성에 따른 참조 데이터 조회
        
        Args:
            age_group: 나이 그룹
            occupation: 직업 분류
            region: 지역 분류
            income_level: 소득 수준
            gender: 성별
        
        Returns:
            List[Dict[str, Any]]: 참조 데이터 리스트
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT rd.*, tc.name as category_name
            FROM reference_data rd
            JOIN transaction_categories tc ON rd.category_id = tc.id
            WHERE rd.age_group = %s
            AND rd.occupation = %s
            AND rd.region = %s
            AND rd.income_level = %s
            AND rd.gender = %s
            ORDER BY rd.spending_ratio DESC
            """,
            (age_group, occupation, region, income_level, gender)
        )
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def create_table(self) -> None:
        """
        참조 데이터 테이블을 생성 (존재하지 않는 경우)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reference_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age_group VARCHAR(20) NOT NULL,
            occupation VARCHAR(50) NOT NULL,
            region VARCHAR(50) NOT NULL,
            income_level VARCHAR(20) NOT NULL,
            gender VARCHAR(10) NOT NULL,
            category_id INT NOT NULL,
            spending_ratio DECIMAL(5,4) NOT NULL,
            avg_amount DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES transaction_categories(id),
            INDEX idx_demographic (age_group, occupation, region, income_level, gender)
        )
        """)
        self.connection.commit()
        cursor.close()