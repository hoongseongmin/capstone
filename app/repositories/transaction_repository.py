# 파일 위치: app/repositories/transaction_repository.py
# 설명: 거래 정보에 대한 데이터베이스 접근 계층 클래스

from typing import List, Optional, Dict, Any
from datetime import datetime
from mysql.connector import MySQLConnection
from app.models.transaction import Transaction

class TransactionRepository:
    """
    거래 데이터에 접근하기 위한 데이터 접근 계층
    
    이 클래스는 거래 정보에 대한 CRUD 작업을 수행합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        TransactionRepository 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
    
    def create(self, transaction: Transaction) -> int:
        """
        새로운 거래를 데이터베이스에 추가
        
        Args:
            transaction: 생성할 거래 객체
        
        Returns:
            int: 생성된 거래의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO transactions 
            (user_id, category_id, amount, transaction_date, payment_method, description) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                transaction.user_id,
                transaction.category_id,
                transaction.amount,
                transaction.transaction_date,
                transaction.payment_method,
                transaction.description
            )
        )
        self.connection.commit()
        transaction_id = cursor.lastrowid
        cursor.close()
        return transaction_id
    
    def create_many(self, transactions: List[Transaction]) -> int:
        """
        여러 거래를 한 번에 데이터베이스에 추가
        
        Args:
            transactions: 생성할 거래 객체 리스트
        
        Returns:
            int: 생성된 거래의 수
        """
        cursor = self.connection.cursor()
        values = [
            (
                t.user_id, 
                t.category_id, 
                t.amount, 
                t.transaction_date, 
                t.payment_method, 
                t.description
            ) 
            for t in transactions
        ]
        
        cursor.executemany(
            """
            INSERT INTO transactions 
            (user_id, category_id, amount, transaction_date, payment_method, description) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            values
        )
        self.connection.commit()
        count = cursor.rowcount
        cursor.close()
        return count
    
    def get_all_by_user_id(self, user_id: int) -> List[Dict[str, Any]]:
        """
        특정 사용자의 모든 거래 내역을 가져옴
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            List[Dict[str, Any]]: 거래 정보 딕셔너리 리스트
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT t.*, c.name as category_name 
            FROM transactions t
            JOIN transaction_categories c ON t.category_id = c.id
            WHERE t.user_id = %s
            ORDER BY t.transaction_date DESC
            """, 
            (user_id,)
        )
        results = cursor.fetchall()
        cursor.close()
        return results
    
    def get_by_id(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """
        지정된 ID의 거래를 가져옴
        
        Args:
            transaction_id: 찾을 거래 ID
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 거래 정보 딕셔너리, 찾지 못한 경우 None
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT t.*, c.name as category_name 
            FROM transactions t
            JOIN transaction_categories c ON t.category_id = c.id
            WHERE t.id = %s
            """, 
            (transaction_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def update(self, transaction_id: int, transaction_data: Dict[str, Any]) -> bool:
        """
        지정된 ID의 거래 정보를 업데이트
        
        Args:
            transaction_id: 업데이트할 거래 ID
            transaction_data: 업데이트할 데이터 딕셔너리
        
        Returns:
            bool: 업데이트 성공 여부
        """
        allowed_fields = {'category_id', 'amount', 'transaction_date', 'payment_method', 'description'}
        update_fields = {k: v for k, v in transaction_data.items() if k in allowed_fields}
        
        if not update_fields:
            return False
        
        set_clause = ", ".join([f"{field} = %s" for field in update_fields.keys()])
        values = list(update_fields.values())
        values.append(transaction_id)
        
        cursor = self.connection.cursor()
        cursor.execute(
            f"UPDATE transactions SET {set_clause} WHERE id = %s",
            values
        )
        self.connection.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        return updated
    
    def delete(self, transaction_id: int) -> bool:
        """
        지정된 ID의 거래를 삭제
        
        Args:
            transaction_id: 삭제할 거래 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM transactions WHERE id = %s", (transaction_id,))
        self.connection.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    
    def get_summary_by_category(self, user_id: int, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        특정 사용자의 카테고리별 거래 요약 정보를 가져옴
        
        Args:
            user_id: 사용자 ID
            start_date: 시작 날짜 (None인 경우 제한 없음)
            end_date: 종료 날짜 (None인 경우 제한 없음)
        
        Returns:
            List[Dict[str, Any]]: 카테고리별 거래 요약 정보
        """
        cursor = self.connection.cursor(dictionary=True)
        
        query = """
        SELECT 
            c.id as category_id, 
            c.name as category_name, 
            SUM(t.amount) as total_amount,
            COUNT(*) as transaction_count,
            MIN(t.transaction_date) as first_transaction,
            MAX(t.transaction_date) as last_transaction
        FROM 
            transactions t
        JOIN 
            transaction_categories c ON t.category_id = c.id
        WHERE 
            t.user_id = %s
        """
        
        params = [user_id]
        
        if start_date:
            query += " AND t.transaction_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND t.transaction_date <= %s"
            params.append(end_date)
        
        query += " GROUP BY c.id, c.name ORDER BY total_amount DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return results
    
    def create_table(self) -> None:
        """
        거래 테이블을 생성 (존재하지 않는 경우)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            category_id INT NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            transaction_date DATETIME NOT NULL,
            payment_method VARCHAR(50),
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (category_id) REFERENCES transaction_categories(id) ON DELETE RESTRICT
        )
        """)
        self.connection.commit()
        cursor.close()