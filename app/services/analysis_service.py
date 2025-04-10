# app/services/analysis_service.py
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from mysql.connector import MySQLConnection
from statistics import mean, stdev

from app.repositories.transaction_repository import TransactionRepository
from app.repositories.user_repository import UserRepository

class AnalysisService:
    """
    소비 패턴 분석 관련 비즈니스 로직을 처리하는 서비스 클래스
    
    이 클래스는 동일 집단 소비습관 분석과 관련된 기능을 제공합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        AnalysisService 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.connection = connection
        self.transaction_repository = TransactionRepository(connection)
        self.user_repository = UserRepository(connection)
    
    def analyze_user_spending_pattern(self, user_id: int, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        특정 사용자의 소비 패턴을 분석
        
        Args:
            user_id: 사용자 ID
            start_date: 시작 날짜 (선택적)
            end_date: 종료 날짜 (선택적)
        
        Returns:
            Dict[str, Any]: 소비 패턴 분석 결과
        """
        # 카테고리별 요약 정보 가져오기
        category_summary = self.transaction_repository.get_summary_by_category(user_id, start_date, end_date)
        
        # 총 지출 계산
        total_spending = sum(item['total_amount'] for item in category_summary)
        
        # 카테고리별 비율 계산
        for item in category_summary:
            item['percentage'] = round((item['total_amount'] / total_spending) * 100, 2) if total_spending > 0 else 0
        
        return {
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date,
            "total_spending": total_spending,
            "category_breakdown": category_summary
        }
    
    def compare_with_group(self, user_id: int, criteria: str) -> Dict[str, Any]:
        """
        동일 그룹 내 다른 사용자들과 소비 패턴 비교
        
        Args:
            user_id: 사용자 ID
            criteria: 비교 기준 (나이, 직업, 지역, 소득 등)
        
        Returns:
            Dict[str, Any]: 그룹 비교 분석 결과
        """
        # 사용자 정보 가져오기
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return {"error": "사용자를 찾을 수 없습니다."}
        
        # 같은 그룹의 사용자 ID 목록 가져오기
        group_user_ids = self._get_same_group_users(user, criteria)
        
        # 사용자의 카테고리별 지출 가져오기
        user_spending = self.transaction_repository.get_summary_by_category(user_id)
        
        # 그룹의 평균 지출 계산
        group_spending = self._calculate_group_average_spending(group_user_ids)
        
        # 비교 결과 생성
        comparison_result = []
        for user_cat in user_spending:
            group_cat = next((item for item in group_spending if item['category_id'] == user_cat['category_id']), None)
            
            if group_cat:
                diff_percentage = ((user_cat['total_amount'] - group_cat['avg_amount']) / group_cat['avg_amount']) * 100 if group_cat['avg_amount'] > 0 else 0
                
                comparison_result.append({
                    "category_id": user_cat['category_id'],
                    "category_name": user_cat['category_name'],
                    "user_amount": user_cat['total_amount'],
                    "group_avg_amount": group_cat['avg_amount'],
                    "diff_percentage": round(diff_percentage, 2),
                    "diff_status": "높음" if diff_percentage > 15 else ("낮음" if diff_percentage < -15 else "평균")
                })
        
        return {
            "user_id": user_id,
            "criteria": criteria,
            "criteria_value": getattr(user, criteria, "Unknown"),
            "group_size": len(group_user_ids),
            "comparison": comparison_result
        }
    
    def _get_same_group_users(self, user: Any, criteria: str) -> List[int]:
        """
        주어진 기준에 따라 같은 그룹에 속하는 사용자 ID 목록을 가져옴
        
        Args:
            user: 사용자 객체
            criteria: 비교 기준 (나이, 직업, 지역, 소득 등)
        
        Returns:
            List[int]: 같은 그룹 사용자 ID 목록
        """
        cursor = self.connection.cursor(dictionary=True)
        
        if criteria == 'age':
            # 나이대별 그룹화 (10세 단위)
            age_group_start = (user.age // 10) * 10
            age_group_end = age_group_start + 9
            cursor.execute(
                "SELECT id FROM users WHERE age BETWEEN %s AND %s AND id != %s",
                (age_group_start, age_group_end, user.id)
            )
        elif criteria == 'job':
            # 같은 직업군
            cursor.execute(
                "SELECT id FROM users WHERE occupation = %s AND id != %s",
                (user.occupation, user.id)
            )
        elif criteria == 'region':
            # 같은 거주 지역
            cursor.execute(
                "SELECT id FROM users WHERE address = %s AND id != %s",
                (user.address, user.id)
            )
        elif criteria == 'income':
            # 같은 소득 수준
            cursor.execute(
                "SELECT id FROM users WHERE income_level = %s AND id != %s",
                (user.income_level, user.id)
            )
        elif criteria == 'gender':
            # 같은 성별
            cursor.execute(
                "SELECT id FROM users WHERE gender = %s AND id != %s",
                (user.gender, user.id)
            )
        else:
            # 기본: 전체 사용자
            cursor.execute("SELECT id FROM users WHERE id != %s", (user.id,))
        
        result = cursor.fetchall()
        cursor.close()
        
        return [item['id'] for item in result]
    
    def _calculate_group_average_spending(self, user_ids: List[int]) -> List[Dict[str, Any]]:
        """
        그룹 내 사용자들의 카테고리별 평균 지출 계산
        
        Args:
            user_ids: 사용자 ID 목록
        
        Returns:
            List[Dict[str, Any]]: 카테고리별 평균 지출 목록
        """
        if not user_ids:
            return []
        
        user_id_placeholders = ', '.join(['%s'] * len(user_ids))
        
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            f"""
            SELECT 
                c.id as category_id, 
                c.name as category_name, 
                AVG(t.amount) as avg_amount,
                COUNT(DISTINCT t.user_id) as user_count
            FROM 
                transactions t
            JOIN 
                transaction_categories c ON t.category_id = c.id
            WHERE 
                t.user_id IN ({user_id_placeholders})
            GROUP BY 
                c.id, c.name
            """,
            user_ids
        )
        
        result = cursor.fetchall()
        cursor.close()
        
        return result
    
    def analyze_monthly_trend(self, user_id: int, months: int = 6) -> Dict[str, Any]:
        """
        특정 사용자의 월별 소비 추세를 분석
        
        Args:
            user_id: 사용자 ID
            months: 조회할 월 수 (기본값: 6)
        
        Returns:
            Dict[str, Any]: 월별 소비 추세 분석 결과
        """
        # 현재 날짜 기준으로 과거 N개월 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT 
                DATE_FORMAT(transaction_date, '%Y-%m') as month,
                SUM(amount) as total_amount,
                COUNT(*) as transaction_count
            FROM 
                transactions
            WHERE 
                user_id = %s AND transaction_date BETWEEN %s AND %s
            GROUP BY 
                DATE_FORMAT(transaction_date, '%Y-%m')
            ORDER BY 
                month ASC
            """,
            (user_id, start_date, end_date)
        )
        
        monthly_totals = cursor.fetchall()
        
        # 카테고리별 월별 지출 계산
        cursor.execute(
            """
            SELECT 
                DATE_FORMAT(t.transaction_date, '%Y-%m') as month,
                c.id as category_id,
                c.name as category_name,
                SUM(t.amount) as total_amount
            FROM 
                transactions t
            JOIN 
                transaction_categories c ON t.category_id = c.id
            WHERE 
                t.user_id = %s AND t.transaction_date BETWEEN %s AND %s
            GROUP BY 
                DATE_FORMAT(t.transaction_date, '%Y-%m'), c.id, c.name
            ORDER BY 
                month ASC, c.name ASC
            """,
            (user_id, start_date, end_date)
        )
        
        category_monthly = cursor.fetchall()
        cursor.close()
        
        # 결과 구성
        result = {
            "user_id": user_id,
            "months_analyzed": months,
            "monthly_totals": monthly_totals,
            "category_monthly": category_monthly
        }
        
        # 증감률 계산 (전월 대비)
        if len(monthly_totals) > 1:
            growth_rates = []
            for i in range(1, len(monthly_totals)):
                prev_amount = monthly_totals[i-1]['total_amount']
                curr_amount = monthly_totals[i]['total_amount']
                growth_rate = ((curr_amount - prev_amount) / prev_amount) * 100 if prev_amount > 0 else 0
                growth_rates.append({
                    "month": monthly_totals[i]['month'],
                    "growth_rate": round(growth_rate, 2)
                })
            result["monthly_growth_rates"] = growth_rates
        
        return result
    
    def detect_anomalies(self, user_id: int) -> Dict[str, Any]:
        """
        특정 사용자의 이상 소비 패턴을 감지
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            Dict[str, Any]: 이상 소비 감지 결과
        """
        # 최근 6개월간 데이터로 이상치 감지
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 약 6개월
        
        # 카테고리별 월별 지출 데이터 가져오기
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT 
                DATE_FORMAT(t.transaction_date, '%Y-%m') as month,
                c.id as category_id,
                c.name as category_name,
                SUM(t.amount) as total_amount
            FROM 
                transactions t
            JOIN 
                transaction_categories c ON t.category_id = c.id
            WHERE 
                t.user_id = %s AND t.transaction_date BETWEEN %s AND %s
            GROUP BY 
                DATE_FORMAT(t.transaction_date, '%Y-%m'), c.id, c.name
            ORDER BY 
                c.name ASC, month ASC
            """,
            (user_id, start_date, end_date)
        )
        
        raw_data = cursor.fetchall()
        cursor.close()
        
        # 카테고리별로 데이터 정리
        category_data = {}
        for item in raw_data:
            cat_id = item['category_id']
            if cat_id not in category_data:
                category_data[cat_id] = {
                    'name': item['category_name'],
                    'amounts': [],
                    'months': []
                }
            category_data[cat_id]['amounts'].append(item['total_amount'])
            category_data[cat_id]['months'].append(item['month'])
        
        # 각 카테고리별로 이상치 감지
        anomalies = []
        for cat_id, data in category_data.items():
            if len(data['amounts']) >= 3:  # 최소 3개 데이터 포인트 필요
                try:
                    # 평균과 표준편차 계산
                    avg = mean(data['amounts'])
                    std = stdev(data['amounts'])
                    
                    # 이상치 기준: 평균 ± 2*표준편차
                    upper_bound = avg + 2 * std
                    lower_bound = avg - 2 * std
                    
                    for i, amount in enumerate(data['amounts']):
                        if amount > upper_bound:
                            anomalies.append({
                                'category_id': cat_id,
                                'category_name': data['name'],
                                'month': data['months'][i],
                                'amount': amount,
                                'average': round(avg, 2),
                                'type': '비정상적 증가',
                                'deviation_percentage': round(((amount - avg) / avg) * 100, 2)
                            })
                        elif amount < lower_bound and lower_bound > 0:
                            anomalies.append({
                                'category_id': cat_id,
                                'category_name': data['name'],
                                'month': data['months'][i],
                                'amount': amount,
                                'average': round(avg, 2),
                                'type': '비정상적 감소',
                                'deviation_percentage': round(((amount - avg) / avg) * 100, 2)
                            })
                except:
                    # 표준편차 계산 실패 시 무시
                    pass
        
        return {
            "user_id": user_id,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies)
        }
    
    def generate_insights(self, user_id: int) -> Dict[str, Any]:
        """
        특정 사용자의 소비 패턴에 대한 인사이트 생성
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            Dict[str, Any]: 소비 인사이트
        """
        # 사용자 정보 가져오기
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return {"error": "사용자를 찾을 수 없습니다."}
        
        insights = []
        
        # 1. 최근 6개월 카테고리별 지출 요약
        spending_pattern = self.analyze_user_spending_pattern(
            user_id, 
            start_date=datetime.now() - timedelta(days=180)
        )
        
        # 2. 주요 소비 카테고리 (상위 3개)
        top_categories = sorted(
            spending_pattern['category_breakdown'], 
            key=lambda x: x['total_amount'], 
            reverse=True
        )[:3]
        
        insights.append({
            "type": "top_spending_categories",
            "title": "주요 소비 카테고리",
            "description": f"최근 6개월 동안 가장 많이 지출한 카테고리는 {', '.join([cat['category_name'] for cat in top_categories])}입니다.",
            "data": top_categories
        })
        
        # 3. 동일 그룹 비교
        group_comparisons = []
        for criteria in ['age', 'job', 'region', 'income']:
            comparison = self.compare_with_group(user_id, criteria)
            if 'error' not in comparison:
                significant_diffs = [
                    item for item in comparison['comparison'] 
                    if abs(item['diff_percentage']) > 20
                ]
                
                if significant_diffs:
                    group_comparisons.append({
                        "criteria": criteria,
                        "significant_differences": significant_diffs
                    })
        
        if group_comparisons:
            insights.append({
                "type": "group_comparison",
                "title": "동일 집단 대비 특이사항",
                "description": "비슷한 인구통계학적 특성을 가진 사용자들과 비교했을 때 다음과 같은 특이사항이 있습니다.",
                "data": group_comparisons
            })
        
        # 4. 이상 소비 감지
        anomalies = self.detect_anomalies(user_id)
        if anomalies['anomaly_count'] > 0:
            insights.append({
                "type": "anomalies",
                "title": "이상 소비 패턴",
                "description": f"최근 6개월 동안 {anomalies['anomaly_count']}건의 이상 소비 패턴이 감지되었습니다.",
                "data": anomalies['anomalies']
            })
        
        # 5. 월별 소비 추세
        monthly_trend = self.analyze_monthly_trend(user_id)
        if 'monthly_growth_rates' in monthly_trend and monthly_trend['monthly_growth_rates']:
            recent_growth = monthly_trend['monthly_growth_rates'][-1]['growth_rate']
            trend_description = "증가하고" if recent_growth > 5 else ("감소하고" if recent_growth < -5 else "유지되고")
            
            insights.append({
                "type": "monthly_trend",
                "title": "소비 추세",
                "description": f"최근 월 기준으로 총 지출이 전월 대비 {abs(recent_growth)}% {trend_description} 있습니다.",
                "data": monthly_trend['monthly_growth_rates']
            })
        
        return {
            "user_id": user_id,
            "user_name": user.name,
            "generated_date": datetime.now(),
            "insights": insights
        }