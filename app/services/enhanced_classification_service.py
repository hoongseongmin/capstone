# app/services/enhanced_classification_service.py
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from collections import Counter

class EnhancedClassificationService:
    """
    통합 거래 분류 및 분석 서비스
    CSV/Excel 파일 업로드부터 고도화된 분류까지 원스톱 처리
    """
    
    def __init__(self):
        # 기본 카테고리 매핑 (NH농협 실제 데이터 기반으로 확장)
        self.CATEGORY_PATTERNS = {
            "식비": {
                "exact_matches": ["맥도날드", "버거킹", "롯데리아", "스타벅스", "이디야", "투썸플레이스", "파리바게뜨"],
                "contains": ["식당", "음식점", "카페", "커피", "치킨", "피자", "도시락", "김밥", "분식", "족발", "보쌈", "삼겹살", "카카오페이", "다모데이토", "이순례", "SOOP"],
                "patterns": [r".*[한중일양]식.*", r".*치킨.*", r".*피자.*", r".*카페.*", r".*커피.*"],
                "exclude": ["커피머신", "커피원두"]
            },
            "교통비": {
                "exact_matches": ["카카오택시", "우버", "타다"],
                "contains": ["택시", "버스", "지하철", "주유소", "SK에너지", "GS칼텍스", "S-OIL", "현대오일뱅크", "파란해충", "케이뱅크"],
                "patterns": [r".*주유.*", r".*택시.*", r".*버스.*", r".*철도.*"],
                "exclude": []
            },
            "주거비": {
                "exact_matches": ["한국전력공사", "서울도시가스", "인천도시가스"],
                "contains": ["전력", "가스", "수도", "관리사무소", "아파트", "빌라", "원룸"],
                "patterns": [r".*전력.*", r".*가스.*", r".*수도.*", r".*관리비.*"],
                "exclude": []
            },
            "통신비": {
                "exact_matches": ["SKT", "KT", "LGU+"],
                "contains": ["통신", "텔레콤", "인터넷", "휴대폰", "핸드폰"],
                "patterns": [r"SK.*통신.*", r"KT.*", r"LG.*통신.*"],
                "exclude": []
            },
            "의료비": {
                "exact_matches": [],
                "contains": ["병원", "의원", "클리닉", "치과", "한의원", "약국", "정형외과", "내과", "피부과"],
                "patterns": [r".*병원.*", r".*의원.*", r".*치과.*", r".*약국.*"],
                "exclude": []
            },
            "교육비": {
                "exact_matches": [],
                "contains": ["학원", "교습소", "과외", "어학원", "학교", "대학교"],
                "patterns": [r".*학원.*", r".*교육.*", r".*학습.*"],
                "exclude": []
            },
            "생활용품비": {
                "exact_matches": ["이마트", "롯데마트", "홈플러스", "코스트코", "GS25", "CU", "세븐일레븐"],
                "contains": ["마트", "편의점", "슈퍼", "대형마트", "하이퍼마켓"],
                "patterns": [r".*마트.*", r".*편의점.*"],
                "exclude": []
            },
            "이미용/화장품": {
                "exact_matches": [],
                "contains": ["미용실", "헤어샵", "네일", "피부관리실", "화장품", "코스메틱"],
                "patterns": [r".*미용.*", r".*헤어.*", r".*뷰티.*"],
                "exclude": []
            },
            "여가비": {
                "exact_matches": ["CGV", "롯데시네마", "메가박스"],
                "contains": ["영화관", "노래방", "PC방", "당구장", "볼링장", "헬스장", "피트니스"],
                "patterns": [r".*PC방.*", r".*노래방.*", r".*헬스.*"],
                "exclude": []
            }
        }
        
        # 거래 패턴 분석을 위한 정규식
        self.TRANSACTION_PATTERNS = {
            "온라인쇼핑": [r".*온라인.*", r".*인터넷.*", r".*쇼핑몰.*", r".*11번가.*", r".*쿠팡.*", r".*옥션.*"],
            "배달주문": [r".*배달.*", r".*요기요.*", r".*배민.*", r".*딜리버리.*"],
            "정기결제": [r".*구독.*", r".*월정액.*", r".*자동결제.*", r".*멤버십.*"],
            "현금인출": [r".*ATM.*", r".*현금인출.*", r".*출금.*"],
            "계좌이체": [r".*이체.*", r".*송금.*", r".*입금.*"]
        }
        
        self.model_initialized = False
    
    def initialize_model(self):
        """고도화된 분류 모델 초기화"""
        if not self.model_initialized:
            print("🚀 통합 분류 서비스 초기화 완료")
            self.model_initialized = True
    
    def analyze_file_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        CSV/Excel 파일 구조 자동 분석
        
        Args:
            df: 파싱된 DataFrame
            
        Returns:
            Dict: 파일 구조 분석 결과
        """
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "detected_columns": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # 컬럼 타입 자동 감지
        column_mapping = {
            "date": ["날짜", "거래일", "거래일시", "일자", "date", "transaction_date"],
            "amount": ["금액", "출금액", "입금액", "출금금액", "입금금액", "사용금액", "결제금액", "amount", "price"],
            "merchant": ["가맹점", "가맹점명", "상호명", "적요", "거래처", "상점명", "merchant", "store"],
            "description": ["거래기록사항", "내역", "설명", "비고", "메모", "description", "memo"],
            "category": ["카테고리", "분류", "category", "type"],
            "payment_method": ["결제수단", "결제방법", "카드", "현금", "payment_method"]
        }
        
        # 일반적인 컬럼 매칭
        for col_type, keywords in column_mapping.items():
            for col in df.columns:
                col_str = str(col).strip()
                col_lower = col_str.lower()
                
                for keyword in keywords:
                    if keyword.lower() in col_lower:
                        analysis["detected_columns"][col_type] = col_str
                        print(f"✅ {col_type} 컬럼 발견: '{col_str}' (키워드: '{keyword}')")
                        break
                if col_type in analysis["detected_columns"]:
                    break
        
        # 데이터 품질 분석
        for col in df.columns:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            analysis["data_quality"][col] = {
                "null_count": int(null_count),
                "null_percentage": round((null_count / len(df)) * 100, 2),
                "unique_count": int(unique_count),
                "sample_values": df[col].dropna().head(3).tolist()
            }
        
        # 권장사항 생성
        if "amount" not in analysis["detected_columns"]:
            analysis["recommendations"].append("⚠️ 금액 컬럼을 찾을 수 없습니다. 수동으로 지정해주세요.")
        
        if "merchant" not in analysis["detected_columns"]:
            analysis["recommendations"].append("⚠️ 가맹점명 컬럼을 찾을 수 없습니다. 거래기록사항 컬럼을 확인해주세요.")
        
        # 디버깅 정보 출력
        print("🔍 디버깅: 감지된 모든 컬럼:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: '{col}' (타입: {type(col)})")

        print("🔍 디버깅: 컬럼 매칭 결과:")
        for col_type, detected_col in analysis["detected_columns"].items():
            print(f"  {col_type}: '{detected_col}'")

        return analysis
    
    def extract_merchant_from_description(self, description: str) -> str:
        """
        거래기록사항에서 가맹점명 추출
        
        Args:
            description: 거래기록사항 텍스트
            
        Returns:
            str: 추출된 가맹점명
        """
        if not description or pd.isna(description):
            return "미상"
        
        desc = str(description).strip()
        
        # 일반적인 거래기록사항 패턴들
        patterns = [
            # "승인 가맹점명 1234" 형태
            r'승인\s+([가-힣A-Za-z0-9\s]+?)\s+\d+',
            # "카드결제 가맹점명" 형태  
            r'카드결제\s+([가-힣A-Za-z0-9\s]+?)(?:\s|$)',
            # "체크결제 가맹점명" 형태
            r'체크결제\s+([가-힣A-Za-z0-9\s]+?)(?:\s|$)',
            # "간편결제 가맹점명" 형태
            r'간편결제\s+([가-힣A-Za-z0-9\s]+?)(?:\s|$)',
            # 일반적인 한글 상호명 (2글자 이상)
            r'([가-힣]{2,}(?:[가-힣A-Za-z0-9\s]*[가-힣A-Za-z0-9])?)',
            # 영문 브랜드명
            r'([A-Za-z]{3,}(?:\s+[A-Za-z]+)*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, desc)
            if matches:
                merchant = matches[0].strip()
                # 너무 짧거나 숫자만 있는 경우 제외
                if len(merchant) >= 2 and not merchant.isdigit():
                    # 불필요한 접미사 제거
                    merchant = re.sub(r'(주식회사|㈜|\(주\)|LTD|CO\.|INC)$', '', merchant).strip()
                    return merchant
        
        # 패턴 매칭 실패시 원본에서 의미있는 부분 추출
        cleaned = re.sub(r'[^\w가-힣\s]', ' ', desc)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) >= 2]
        
        if words:
            return words[0]
        
        return desc[:20] if len(desc) > 20 else desc
    
    def advanced_categorize(self, merchant_name: str, description: str = "", amount: float = 0) -> Dict[str, Any]:
        """
        고도화된 카테고리 분류 (다중 정보 활용)
        
        Args:
            merchant_name: 가맹점명
            description: 거래기록사항
            amount: 거래금액
            
        Returns:
            Dict: 분류 결과 및 신뢰도
        """
        if not merchant_name or str(merchant_name).strip() == '':
            return {"category": "기타", "confidence": 0, "method": "default"}
        
        merchant = str(merchant_name).lower().strip()
        desc = str(description).lower().strip() if description else ""
        
        # 1차: 정확한 매칭
        for category, patterns in self.CATEGORY_PATTERNS.items():
            # 정확한 이름 매칭
            for exact in patterns["exact_matches"]:
                if exact.lower() in merchant:
                    return {"category": category, "confidence": 95, "method": "exact_match", "matched": exact}
            
            # 제외 키워드 체크
            if any(exclude.lower() in merchant for exclude in patterns["exclude"]):
                continue
            
            # 포함 키워드 매칭
            for keyword in patterns["contains"]:
                if keyword.lower() in merchant or keyword.lower() in desc:
                    return {"category": category, "confidence": 85, "method": "keyword_match", "matched": keyword}
            
            # 정규식 패턴 매칭
            for pattern in patterns["patterns"]:
                if re.search(pattern, merchant) or re.search(pattern, desc):
                    return {"category": category, "confidence": 75, "method": "pattern_match", "matched": pattern}
        
        # 2차: 금액 기반 추론 (특정 패턴들)
        if amount > 0:
            if 10000 <= amount <= 50000 and any(word in merchant for word in ["주유", "기름", "gas"]):
                return {"category": "교통비", "confidence": 70, "method": "amount_inference"}
            
            if amount >= 100000 and any(word in merchant for word in ["전력", "가스", "관리"]):
                return {"category": "주거비", "confidence": 70, "method": "amount_inference"}
            
            if 3000 <= amount <= 15000 and any(word in merchant for word in ["편의점", "gs25", "cu", "세븐"]):
                return {"category": "생활용품비", "confidence": 70, "method": "amount_inference"}
        
        # 3차: 거래 패턴 분석
        full_text = f"{merchant} {desc}"
        for pattern_type, patterns in self.TRANSACTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_text):
                    if pattern_type == "온라인쇼핑":
                        return {"category": "생활용품비", "confidence": 60, "method": "transaction_pattern"}
                    elif pattern_type == "배달주문":
                        return {"category": "식비", "confidence": 80, "method": "transaction_pattern"}
        
        return {"category": "기타", "confidence": 0, "method": "unclassified"}
    
    def process_transactions_with_analysis(self, df: pd.DataFrame, file_structure: Dict) -> Dict[str, Any]:
        """
        거래 데이터 처리 및 고도화된 분석
        
        Args:
            df: 거래 데이터 DataFrame
            file_structure: 파일 구조 분석 결과
            
        Returns:
            Dict: 처리된 거래 데이터 및 분석 결과
        """
        detected_cols = file_structure["detected_columns"]
        
        # NH농협 파일 특별 처리: 컬럼명이 Unnamed인 경우 자동 매핑
        if not detected_cols.get("amount") and any("Unnamed" in str(col) for col in df.columns):
            print("🔧 NH농협 파일 특별 처리 모드 활성화")
            
            # 데이터에서 실제 값들을 확인하여 컬럼 추정
            for idx, col in enumerate(df.columns):
                sample_values = df[col].dropna().head(5).tolist()
                print(f"컬럼 {idx} ({col}) 샘플: {sample_values}")
                
                # 금액 컬럼 찾기 (숫자이고 1000 이상인 값들이 많은 컬럼)
                numeric_values = []
                for val in sample_values:
                    try:
                        # 쉼표 제거 후 숫자 변환 시도
                        val_str = str(val).replace(",", "").strip()
                        if val_str and val_str != 'nan':
                            num_val = float(val_str)
                            if num_val > 0:
                                numeric_values.append(num_val)
                    except (ValueError, TypeError):
                        pass
                
                # 유효한 금액 데이터가 있고, 1000 이상의 값이 있으면 금액 컬럼으로 추정
                if len(numeric_values) >= 2 and any(v >= 1000 for v in numeric_values):
                    detected_cols["amount"] = col
                    print(f"✅ 금액 컬럼으로 추정: {col} (샘플: {numeric_values})")
                    break
            
            # 가맹점명/거래기록사항 컬럼 찾기 (한글이 포함된 텍스트 컬럼)
            if not detected_cols.get("description"):
                for idx, col in enumerate(df.columns):
                    sample_values = df[col].dropna().head(5).tolist()
                    korean_count = 0
                    for val in sample_values:
                        val_str = str(val)
                        if re.search(r'[가-힣]', val_str) and len(val_str) >= 2:
                            korean_count += 1
                    
                    if korean_count >= 2:
                        detected_cols["description"] = col
                        detected_cols["merchant"] = col  # 같은 컬럼에서 가맹점명 추출
                        print(f"✅ 거래기록사항 컬럼으로 추정: {col}")
                        break
        
        # 필수 컬럼 확인
        amount_col = detected_cols.get("amount")
        merchant_col = detected_cols.get("merchant")
        description_col = detected_cols.get("description")
        date_col = detected_cols.get("date")
        
        if not amount_col:
            raise ValueError("금액 컬럼을 찾을 수 없습니다.")
        
        processed_transactions = []
        classification_stats = Counter()
        confidence_distribution = Counter()
        amount_by_category = {}
        
        for index, row in df.iterrows():
            try:
                # 금액 파싱
                amount_str = str(row[amount_col]).replace(",", "").replace(" ", "").strip()
                if pd.isna(row[amount_col]) or amount_str == '' or amount_str == 'nan':
                    continue
                
                amount = 0
                try:
                    amount = float(re.sub(r'[^\d.]', '', amount_str))
                except (ValueError, TypeError):
                    continue
                
                if amount <= 0:
                    continue
                
                # 가맹점명 추출
                merchant_name = "미상"
                if merchant_col and pd.notna(row[merchant_col]):
                    merchant_name = str(row[merchant_col]).strip()
                elif description_col and pd.notna(row[description_col]):
                    merchant_name = self.extract_merchant_from_description(str(row[description_col]))
                
                # 거래기록사항
                description = ""
                if description_col and pd.notna(row[description_col]):
                    description = str(row[description_col])
                
                # 날짜 파싱
                transaction_date = datetime.now()
                if date_col and pd.notna(row[date_col]):
                    try:
                        transaction_date = pd.to_datetime(row[date_col])
                    except:
                        pass
                
                # 고도화된 분류
                classification_result = self.advanced_categorize(merchant_name, description, amount)
                category = classification_result["category"]
                confidence = classification_result["confidence"]
                
                # 통계 업데이트
                classification_stats[category] += 1
                confidence_range = f"{(confidence//10)*10}-{(confidence//10)*10+9}%"
                confidence_distribution[confidence_range] += 1
                
                if category not in amount_by_category:
                    amount_by_category[category] = 0
                amount_by_category[category] += amount
                
                transaction = {
                    "transaction_date": transaction_date.isoformat() if hasattr(transaction_date, 'isoformat') else str(transaction_date),
                    "amount": amount,
                    "store_name": merchant_name,
                    "category": category,
                    "description": description,
                    "payment_method": "카드",
                    "classification_confidence": confidence,
                    "classification_method": classification_result["method"],
                    "matched_keyword": classification_result.get("matched", "")
                }
                
                processed_transactions.append(transaction)
                
            except Exception as e:
                print(f"행 {index} 처리 중 오류: {e}")
                continue
        
        # 고도화된 분석 결과
        total_amount = sum(amount_by_category.values()) if amount_by_category else 0
        
        analysis_result = {
            "total_transactions": len(processed_transactions),
            "total_amount": total_amount,
            "category_breakdown": {
                cat: {
                    "count": classification_stats[cat],
                    "amount": amount_by_category.get(cat, 0),
                    "percentage": round((amount_by_category.get(cat, 0) / total_amount) * 100, 2) if total_amount > 0 else 0
                }
                for cat in classification_stats.keys()
            },
            "classification_quality": {
                "high_confidence": sum(1 for t in processed_transactions if t["classification_confidence"] >= 80),
                "medium_confidence": sum(1 for t in processed_transactions if 60 <= t["classification_confidence"] < 80),
                "low_confidence": sum(1 for t in processed_transactions if t["classification_confidence"] < 60),
                "confidence_distribution": dict(confidence_distribution)
            },
            "unclassified_analysis": {
                "count": classification_stats.get("기타", 0),
                "percentage": round((classification_stats.get("기타", 0) / len(processed_transactions)) * 100, 2) if processed_transactions else 0,
                "top_unclassified": [
                    t["store_name"] for t in processed_transactions 
                    if t["category"] == "기타" and t["classification_confidence"] == 0
                ][:10]
            }
        }
        
        return {
            "transactions": processed_transactions,
            "analysis": analysis_result,
            "recommendations": self._generate_recommendations(analysis_result)
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        unclassified_pct = analysis["unclassified_analysis"]["percentage"]
        if unclassified_pct > 20:
            recommendations.append(f"⚠️ 기타로 분류된 거래가 {unclassified_pct}%로 높습니다. 추가 키워드 등록을 권장합니다.")
        
        quality = analysis["classification_quality"]
        total_transactions = analysis["total_transactions"]
        if total_transactions > 0:
            low_confidence_pct = round((quality["low_confidence"] / total_transactions) * 100, 2)
            if low_confidence_pct > 30:
                recommendations.append(f"📊 낮은 신뢰도 분류({low_confidence_pct}%)가 많습니다. 거래기록사항 품질 개선이 필요합니다.")
        
        # 카테고리별 이상 패턴 감지
        for category, data in analysis["category_breakdown"].items():
            if data["percentage"] > 50:
                recommendations.append(f"💡 '{category}' 카테고리가 전체의 {data['percentage']}%를 차지합니다. 세부 분류를 고려해보세요.")
        
        return recommendations

# 전역 인스턴스
enhanced_classification_service = EnhancedClassificationService()