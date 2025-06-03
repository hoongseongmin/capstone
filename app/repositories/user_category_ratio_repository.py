# app/repositories/user_category_ratio_repository.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from mysql.connector import MySQLConnection
from app.models.user_category_ratio import UserCategoryRatio

class UserCategoryRatioRepository:
    """
    사용자별 카테고리 지출 비율에 접근하기 위한 데이터 접근 계층
    
    이 클래스는 사용자별 카테고리 지출 비율에 대한 CRUD 작업을 수행합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        UserCategoryRatioRepository 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
    
    def create(self, data: UserCategoryRatio) -> int:
        """
        새로운 사용자별 카테고리 지출 비율을 데이터베이스에 추가
        
        Args:
            data: 생성할 사용자별 카테고리 지출 비율 객체
        
        Returns:
            int: 생성된 데이터의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO user_category_ratios 
            (user_id, category_id, ratio, avg_amount, period_start, period_end) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                data.user_id,
                data.category_id,
                data.ratio,
                data.avg_amount,
                data.period_start,
                data.period_end
            )
        )
        self.connection.commit()
        data_id = cursor.lastrowid
        cursor.close()
        return data_id
    
    def get_by_user_id(self, user_id: int, period_start: Optional[datetime] = None, 
                     period_end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        특정 사용자의 카테고리별 지출 비율을 가져옴
        
        Args:
            user_id: 사용자 ID
            period_start: 시작 날짜 (None인 경우 최근 기간)
            period_end: 종료 날짜 (None인 경우 최근 기간)
        
        Returns:
            List[Dict[str, Any]]: 사용자별 카테고리 지출 비율 리스트
        """
        cursor = self.connection.cursor(dictionary=True)
        
        if period_start is None or period_end is None:
            # 가장 최근 기간 데이터 조회
            cursor.execute(
                """
                SELECT ucr.*, tc.name as category_name
                FROM user_category_ratios ucr
                JOIN transaction_categories tc ON ucr.category_id = tc.id
                WHERE ucr.user_id = %s
                AND (ucr.period_start, ucr.period_end) IN (
                    SELECT period_start, period_end
                    FROM user_category_ratios
                    WHERE user_id = %s
                    ORDER BY period_end DESC
                    LIMIT 1
                )
                ORDER BY ucr.ratio DESC
                """,
                (user_id, user_id)
            )
        else:
            # 특정 기간 데이터 조회
            cursor.execute(
                """
                SELECT ucr.*, tc.name as category_name
                FROM user_category_ratios ucr
                JOIN transaction_categories tc ON ucr.category_id = tc.id
                WHERE ucr.user_id = %s
                AND ucr.period_start = %s
                AND ucr.period_end = %s
                ORDER BY ucr.ratio DESC
                """,
                (user_id, period_start, period_end)
            )
        
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def create_table(self) -> None:
        """
        사용자별 카테고리 지출 비율 테이블을 생성 (존재하지 않는 경우)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_category_ratios (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            category_id INT NOT NULL,
            ratio DECIMAL(5,4) NOT NULL,
            avg_amount DECIMAL(10,2) NOT NULL,
            period_start DATE NOT NULL,
            period_end DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (category_id) REFERENCES transaction_categories(id),
            UNIQUE KEY unq_user_cat_period (user_id, category_id, period_start, period_end)
        )
        """)
        self.connection.commit()
        cursor.close()