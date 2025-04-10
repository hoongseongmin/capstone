# app/services/transaction_service.py
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from mysql.connector import MySQLConnection

from app.models.transaction import Transaction
from app.repositories.transaction_repository import TransactionRepository
from app.repositories.transaction_category_repository import TransactionCategoryRepository

class TransactionService:
    """
    거래 관련 비즈니스 로직을 처리하는 서비스 클래스
    
    이 클래스는 Repository 계층을 사용하여 거래 정보를 관리합니다.
    AI 담당자로부터 전달받은 분석 데이터를 처리합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        TransactionService 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
        self.repository = TransactionRepository(connection)
        self.category_repository = TransactionCategoryRepository(connection)
    
    def create_transaction(self, transaction: Transaction) -> int:
        """
        새로운 거래를 생성
        
        Args:
            transaction: 생성할 거래 객체
        
        Returns:
            int: 생성된 거래의 ID
        """
        return self.repository.create(transaction)
    
    def get_user_transactions(self, user_id: int) -> List[Dict[str, Any]]:
        """
        특정 사용자의 모든 거래 내역을 가져옴
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            List[Dict[str, Any]]: 거래 정보 딕셔너리 리스트
        """
        return self.repository.get_all_by_user_id(user_id)
    
    def get_transaction_by_id(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """
        지정된 ID의 거래를 가져옴
        
        Args:
            transaction_id: 찾을 거래 ID
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 거래 정보 딕셔너리, 찾지 못한 경우 None
        """
        return self.repository.get_by_id(transaction_id)
    
    def update_transaction(self, transaction_id: int, transaction_data: Dict[str, Any]) -> bool:
        """
        지정된 ID의 거래 정보를 업데이트
        
        Args:
            transaction_id: 업데이트할 거래 ID
            transaction_data: 업데이트할 데이터 딕셔너리
        
        Returns:
            bool: 업데이트 성공 여부
        """
        return self.repository.update(transaction_id, transaction_data)
    
    def delete_transaction(self, transaction_id: int) -> bool:
        """
        지정된 ID의 거래를 삭제
        
        Args:
            transaction_id: 삭제할 거래 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        return self.repository.delete(transaction_id)
    
    def get_category_summary(self, user_id: int, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        특정 사용자의 카테고리별 거래 요약 정보를 가져옴
        
        Args:
            user_id: 사용자 ID
            start_date: 시작 날짜 (None인 경우 제한 없음)
            end_date: 종료 날짜 (None인 경우 제한 없음)
        
        Returns:
            List[Dict[str, Any]]: 카테고리별 거래 요약 정보
        """
        return self.repository.get_summary_by_category(user_id, start_date, end_date)
    
    def process_ai_analyzed_transactions(self, user_id: int, transactions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        AI 담당자로부터 전달받은 분석된 거래 데이터를 처리하여 저장
        
        Args:
            user_id: 사용자 ID
            transactions_data: 거래 데이터 리스트 (AI 분석 결과)
                각 항목은 다음 키를 포함해야 함:
                - category: 거래 카테고리 (주거비, 교통비 등)
                - amount: : 거래 금액
                - date: 거래 날짜
                - description: 거래 설명 (선택적)
                - payment_method: 결제 수단 (선택적)
            
        Returns:
            Dict[str, Any]: 처리 결과 (성공 여부, 처리된 거래 수, 오류 메시지 등)
        """
        try:
            # 카테고리 정보 가져오기
            categories = self.category_repository.get_all()
            categories_dict = {cat['name']: cat['id'] for cat in categories}
            
            # 거래 내역 변환
            transactions = []
            errors = []
            success_count = 0
            
            for index, transaction_data in enumerate(transactions_data):
                try:
                    # 필수 필드 확인
                    if 'category' not in transaction_data:
                        errors.append(f"{index+1}번 거래: 카테고리 정보가 없습니다.")
                        continue
                    
                    if 'amount' not in transaction_data:
                        errors.append(f"{index+1}번 거래: 금액 정보가 없습니다.")
                        continue
                    
                    if 'date' not in transaction_data:
                        errors.append(f"{index+1}번 거래: 날짜 정보가 없습니다.")
                        continue
                    
                    # 카테고리 ID 확인
                    category_name = transaction_data['category']
                    if category_name not in categories_dict:
                        # 만약 카테고리가 없으면 '기타'로 설정
                        category_id = categories_dict.get('기타', 1)
                        errors.append(f"{index+1}번 거래: 알 수 없는 카테고리 '{category_name}', '기타'로 설정됨")
                    else:
                        category_id = categories_dict[category_name]
                    
                    # 거래 객체 생성
                    transaction = Transaction(
                        id=None,
                        user_id=user_id,
                        category_id=category_id,
                        amount=float(transaction_data['amount']),
                        transaction_date=transaction_data['date'],
                        payment_method=transaction_data.get('payment_method'),
                        description=transaction_data.get('description', '')
                    )
                    
                    transactions.append(transaction)
                    success_count += 1
                    
                except Exception as e:
                    errors.append(f"{index+1}번 거래: 처리 중 오류 발생 - {str(e)}")
            
            # 데이터베이스에 거래 데이터 추가
            if transactions:
                self.repository.create_many(transactions)
            
            return {
                "success": True,
                "total": len(transactions_data),
                "processed_count": success_count,
                "error_count": len(errors),
                "errors": errors[:10],  # 처음 10개 오류만 반환
                "has_more_errors": len(errors) > 10
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"데이터 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def create_tables(self) -> None:
        """
        필요한 테이블을 생성 (존재하지 않는 경우)
        """
        self.category_repository.create_table()
        self.repository.create_table()
        
        # 기본 카테고리 초기화
        self.category_repository.initialize_default_categories()