# app/services/reference_data_service.py
from typing import List, Dict, Any, Optional
import csv
import io
from datetime import datetime
from mysql.connector import MySQLConnection

from app.models.reference_data import ReferenceData
from app.repositories.reference_data_repository import ReferenceDataRepository
from app.repositories.transaction_category_repository import TransactionCategoryRepository

class ReferenceDataService:
    """
    참조 데이터 관련 비즈니스 로직을 처리하는 서비스 클래스
    
    이 클래스는 동일 집단 소비습관 참조 데이터를 관리합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        ReferenceDataService 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
        self.repository = ReferenceDataRepository(connection)
        self.category_repository = TransactionCategoryRepository(connection)
    
    def import_from_csv(self, csv_content: str) -> Dict[str, Any]:
        """
        CSV 파일에서 참조 데이터를 가져와 데이터베이스에 저장
        
        Args:
            csv_content: CSV 내용 문자열
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 카테고리 정보 가져오기
            categories = self.category_repository.get_all()
            categories_dict = {cat['name']: cat['id'] for cat in categories}
            
            reference_data = []
            errors = []
            success_count = 0
            
            # CSV 파싱
            csv_file = io.StringIO(csv_content)
            csv_reader = csv.DictReader(csv_file)
            
            for row_idx, row in enumerate(csv_reader, start=1):
                try:
                    # 필수 필드 확인
                    required_fields = ['age_group', 'occupation', 'region', 'income_level', 
                                      'gender', 'category', 'spending_ratio', 'avg_amount']
                    
                    for field in required_fields:
                        if field not in row or not row[field]:
                            errors.append(f"{row_idx}번 행: '{field}' 필드가 없거나 비어 있습니다.")
                            continue
                    
                    # 카테고리 ID 확인
                    category_name = row['category']
                    if category_name not in categories_dict:
                        # 카테고리가 없으면 '기타'로 설정
                        category_id = categories_dict.get('기타', 1)
                        errors.append(f"{row_idx}번 행: 알 수 없는 카테고리 '{category_name}', '기타'로 설정됨")
                    else:
                        category_id = categories_dict[category_name]
                    
                    # 참조 데이터 객체 생성
                    ref_data = ReferenceData(
                        id=None,
                        age_group=row['age_group'],
                        occupation=row['occupation'],
                        region=row['region'],
                        income_level=row['income_level'],
                        gender=row['gender'],
                        category_id=category_id,
                        spending_ratio=float(row['spending_ratio']),
                        avg_amount=float(row['avg_amount'])
                    )
                    
                    reference_data.append(ref_data)
                    success_count += 1
                    
                except Exception as e:
                    errors.append(f"{row_idx}번 행: 처리 중 오류 발생 - {str(e)}")
            
            # 데이터베이스에 참조 데이터 추가
            if reference_data:
                self.repository.create_many(reference_data)
            
            return {
                "success": True,
                "total": row_idx if 'row_idx' in locals() else 0,
                "processed_count": success_count,
                "error_count": len(errors),
                "errors": errors[:10],
                "has_more_errors": len(errors) > 10
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"데이터 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def get_reference_data(self, age_group: Optional[str] = None, occupation: Optional[str] = None,
                         region: Optional[str] = None, income_level: Optional[str] = None,
                         gender: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        필터링된 참조 데이터 조회
        
        Args:
            age_group: 연령대 필터 (선택)
            occupation: 직업 필터 (선택)
            region: 지역 필터 (선택)
            income_level: 소득 수준 필터 (선택)
            gender: 성별 필터 (선택)
            
        Returns:
            List[Dict[str, Any]]: 참조 데이터 목록
        """
        cursor = self.connection.cursor(dictionary=True)
        
        query = """
        SELECT rd.*, tc.name as category_name
        FROM reference_data rd
        JOIN transaction_categories tc ON rd.category_id = tc.id
        WHERE 1=1
        """
        
        params = []
        
        if age_group:
            query += " AND rd.age_group = %s"
            params.append(age_group)
            
        if occupation:
            query += " AND rd.occupation = %s"
            params.append(occupation)
            
        if region:
            query += " AND rd.region = %s"
            params.append(region)
            
        if income_level:
            query += " AND rd.income_level = %s"
            params.append(income_level)
            
        if gender:
            query += " AND rd.gender = %s"
            params.append(gender)
        
        query += " ORDER BY rd.spending_ratio DESC"
        
        cursor.execute(query, params)
        result = cursor.fetchall()
        cursor.close()
        
        return result
    
    def create_tables(self) -> None:
        """
        필요한 테이블을 생성 (존재하지 않는 경우)
        """
        self.repository.create_table()