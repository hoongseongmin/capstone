import requests
import re
from typing import Dict, List, Optional
from urllib.parse import quote
import time

class BusinessSearchService:
    """
    사업자등록정보 및 추가 정보 검색 서비스
    """
    
    def __init__(self):
        # 공공 API 키 (실제 사용시 환경변수로 관리)
        self.DATA_GO_KR_API_KEY = None  # 공공데이터포털 API 키
        
        # 카테고리 키워드 매핑 (확장된 버전)
        self.EXTENDED_KEYWORDS = {
            "식비": {
                "keywords": ["식당", "음식", "카페", "커피", "치킨", "피자", "햄버거", "중국집", "일식", "한식", "양식", 
                           "분식", "도시락", "김밥", "라면", "국수", "냉면", "삼겹살", "갈비", "회", "초밥", "파스타",
                           "맥도날드", "버거킹", "롯데리아", "스타벅스", "이디야", "투썸플레이스", "파리바게뜨",
                           "배달", "요기요", "배민", "쿠팡이츠", "베이커리", "제과점"],
                "business_types": ["일반음식점", "휴게음식점", "제과점", "카페"]
            },
            "의료비": {
                "keywords": ["병원", "의원", "클리닉", "치과", "한의원", "약국", "정형외과", "내과", "외과", "산부인과", 
                           "소아과", "피부과", "안과", "이비인후과", "정신과", "재활의학과", "성형외과", "신경과",
                           "비뇨기과", "흉부외과", "응급실", "검진", "건강검진"],
                "business_types": ["의료기관", "약국", "치과의원", "한의원"]
            },
            "이미용/화장품": {
                "keywords": ["미용실", "헤어샵", "네일샵", "피부관리", "마사지", "사우나", "찜질방", "화장품", 
                           "코스메틱", "에스테틱", "스파", "뷰티", "헤어", "펌", "염색", "매니큐어", "페디큐어",
                           "왁싱", "아모레퍼시픽", "LG생활건강", "올리브영", "롭스"],
                "business_types": ["미용업", "이용업", "화장품판매업"]
            },
            "교육비": {
                "keywords": ["학원", "교습소", "과외", "영어", "수학", "국어", "학습", "교육", "어학원", "컴퓨터학원", 
                           "피아노", "음악", "미술", "태권도", "체육", "요가", "필라테스", "학교", "대학교",
                           "유치원", "어린이집", "도서관", "문구점", "교재"],
                "business_types": ["학원", "교습소", "체육시설업"]
            },
            "여가비": {
                "keywords": ["노래방", "PC방", "당구장", "볼링장", "골프", "찜질방", "영화관", "게임", "오락", 
                           "스포츠", "헬스", "피트니스", "수영장", "테니스", "배드민턴", "CGV", "롯데시네마",
                           "메가박스", "놀이공원", "테마파크", "카지노", "경마장"],
                "business_types": ["유흥업소", "체육시설업", "영화상영업", "게임제공업"]
            },
            "쇼핑": {
                "keywords": ["쇼핑몰", "백화점", "마트", "할인점", "의류", "신발", "가방", "액세서리", "잡화", 
                           "전자제품", "가전", "가구", "생활용품", "문구", "서점", "온라인쇼핑", "인터넷쇼핑",
                           "이마트", "롯데마트", "홈플러스", "코스트코", "현대백화점", "롯데백화점"],
                "business_types": ["소매업", "의류판매업", "전자제품판매업", "서점업"]
            },
            "생활용품비": {
                "keywords": ["편의점", "슈퍼마켓", "마트", "대형마트", "하이퍼마켓", "GS25", "CU", "세븐일레븐",
                           "이마트24", "미니스톱", "생필품", "세제", "화장지", "샴푸", "비누"],
                "business_types": ["편의점", "슈퍼마켓", "생활용품점"]
            },
            "교통비": {
                "keywords": ["주유소", "기름", "휘발유", "경유", "SK에너지", "GS칼텍스", "S-OIL", "현대오일뱅크",
                           "택시", "버스", "지하철", "기차", "항공", "렌터카", "톨게이트", "주차장",
                           "카카오택시", "우버", "타다"],
                "business_types": ["주유소", "교통업", "주차장업"]
            },
            "통신비": {
                "keywords": ["통신", "휴대폰", "핸드폰", "인터넷", "전화", "SKT", "KT", "LGU+", "알뜰폰",
                           "데이터", "요금제", "로밍"],
                "business_types": ["통신업"]
            },
            "주거비": {
                "keywords": ["전력", "가스", "수도", "관리비", "임대료", "월세", "전세", "부동산", "아파트",
                           "한국전력공사", "도시가스", "수도공사", "관리사무소"],
                "business_types": ["공공요금", "부동산업"]
            },
            "금융비": {
                "keywords": ["은행", "카드", "대출", "보험", "증권", "투자", "펀드", "적금", "예금",
                           "ATM", "현금인출", "이자", "수수료"],
                "business_types": ["금융업", "보험업"]
            }
        }
    
    def extract_business_keywords(self, store_name: str) -> List[str]:
        """
        상호명에서 사업 관련 키워드 추출
        
        Args:
            store_name: 상호명
            
        Returns:
            List[str]: 추출된 키워드 목록
        """
        if not store_name or str(store_name).strip() == '':
            return []
        
        name = str(store_name).strip()
        keywords = []
        
        # 1. 업종 관련 키워드 추출
        for category, data in self.EXTENDED_KEYWORDS.items():
            for keyword in data["keywords"]:
                if keyword.lower() in name.lower():
                    keywords.append(keyword)
        
        # 2. 일반적인 사업자 키워드 패턴
        business_patterns = [
            r'([가-힣]+)식당',
            r'([가-힣]+)카페',
            r'([가-힣]+)마트',
            r'([가-힣]+)약국',
            r'([가-힣]+)병원',
            r'([가-힣]+)학원',
            r'([가-힣]+)미용실',
            r'([가-힣]+)치과',
            r'([가-힣]+)한의원',
            r'([가-힣]+)PC방',
            r'([가-힣]+)노래방'
        ]
        
        for pattern in business_patterns:
            matches = re.findall(pattern, name)
            keywords.extend(matches)
        
        # 3. 브랜드명/체인점 패턴 추출
        chain_patterns = [
            r'([가-힣A-Za-z]+)\s*(점|매장|지점)',
            r'([가-힣A-Za-z]{2,})\s*[0-9]*호점?',
        ]
        
        for pattern in chain_patterns:
            matches = re.findall(pattern, name)
            if matches:
                keywords.extend([match[0] if isinstance(match, tuple) else match for match in matches])
        
        # 4. 숫자나 특수문자 제거된 핵심 키워드
        clean_name = re.sub(r'[0-9\-_\(\)\[\]{}]', ' ', name)
        words = [word.strip() for word in clean_name.split() if len(word.strip()) >= 2]
        keywords.extend(words)
        
        return list(set(keywords))  # 중복 제거
    
    def search_business_info_mock(self, store_name: str) -> Dict:
        """
        사업자정보 검색 (Mock 버전 - 실제 API 연동 전까지 사용)
        
        Args:
            store_name: 상호명
            
        Returns:
            Dict: 검색 결과
        """
        if not store_name or str(store_name).strip() == '':
            return {
                "store_name": store_name,
                "extracted_keywords": [],
                "suggested_categories": [],
                "search_method": "empty_input",
                "has_external_data": False
            }
        
        keywords = self.extract_business_keywords(store_name)
        
        # 키워드 기반 카테고리 추천
        confidence_scores = {}
        
        for category, data in self.EXTENDED_KEYWORDS.items():
            score = 0
            matched_keywords = []
            
            # 직접 매칭 점수 계산
            for keyword in keywords:
                for category_keyword in data["keywords"]:
                    if keyword.lower() in category_keyword.lower() or category_keyword.lower() in keyword.lower():
                        score += 1
                        if keyword not in matched_keywords:
                            matched_keywords.append(keyword)
                        break
            
            # 상호명에 카테고리 키워드가 직접 포함되는 경우
            store_lower = store_name.lower()
            for category_keyword in data["keywords"]:
                if category_keyword.lower() in store_lower:
                    score += 2  # 직접 매칭은 더 높은 점수
                    if category_keyword not in matched_keywords:
                        matched_keywords.append(category_keyword)
            
            if score > 0:
                confidence_scores[category] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "confidence": min(score * 25, 100)  # 점수당 25%, 최대 100%
                }
        
        # 점수 순으로 정렬
        sorted_categories = sorted(confidence_scores.items(), 
                                 key=lambda x: x[1]["score"], reverse=True)
        
        suggested_categories = [
            {
                "category": cat,
                "confidence": info["confidence"],
                "matched_keywords": info["matched_keywords"],
                "reason": f"{', '.join(info['matched_keywords'])} 키워드 매칭"
            }
            for cat, info in sorted_categories[:3]  # 상위 3개만
        ]
        
        return {
            "store_name": store_name,
            "extracted_keywords": keywords,
            "suggested_categories": suggested_categories,
            "search_method": "keyword_analysis",
            "has_external_data": False
        }
    
    def analyze_unclassified_batch(self, unclassified_transactions: List[Dict]) -> Dict:
        """
        기타로 분류된 거래들을 일괄 분석
        
        Args:
            unclassified_transactions: 기타로 분류된 거래 목록
            
        Returns:
            Dict: 분석 결과
        """
        analysis_results = []
        keyword_frequency = {}
        
        for transaction in unclassified_transactions:
            try:
                store_name = transaction.get('store_name', '')
                if not store_name or store_name == '기타':
                    continue
                
                # 개별 분석
                business_info = self.search_business_info_mock(store_name)
                analysis_results.append({
                    "transaction": transaction,
                    "analysis": business_info
                })
                
                # 키워드 빈도 계산
                for keyword in business_info["extracted_keywords"]:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
                    
            except Exception as e:
                print(f"거래 분석 중 오류: {e}")
                continue
        
        # 빈도 높은 키워드 순으로 정렬
        top_keywords = sorted(keyword_frequency.items(), 
                            key=lambda x: x[1], reverse=True)[:20]
        
        # 카테고리별 재분류 제안
        category_suggestions = {}
        for result in analysis_results:
            for suggestion in result["analysis"]["suggested_categories"]:
                category = suggestion["category"]
                if category not in category_suggestions:
                    category_suggestions[category] = {
                        "count": 0,
                        "transactions": [],
                        "avg_confidence": 0,
                        "total_confidence": 0
                    }
                category_suggestions[category]["count"] += 1
                category_suggestions[category]["transactions"].append(result["transaction"])
                category_suggestions[category]["total_confidence"] += suggestion["confidence"]
        
        # 평균 신뢰도 계산
        for category in category_suggestions:
            if category_suggestions[category]["count"] > 0:
                category_suggestions[category]["avg_confidence"] = round(
                    category_suggestions[category]["total_confidence"] / category_suggestions[category]["count"], 2
                )
        
        return {
            "total_analyzed": len(analysis_results),
            "top_keywords": top_keywords,
            "category_suggestions": category_suggestions,
            "detailed_analysis": analysis_results[:10],  # 상위 10개만 반환 (응답 크기 제한)
            "summary": {
                "most_common_keywords": [kw[0] for kw in top_keywords[:5]],
                "suggested_recategorization": len([
                    cat for cat, data in category_suggestions.items() 
                    if data["avg_confidence"] > 60
                ]),
                "total_unclassified_amount": sum(
                    t.get("amount", 0) for t in unclassified_transactions
                ),
                "avg_transaction_amount": round(
                    sum(t.get("amount", 0) for t in unclassified_transactions) / len(unclassified_transactions), 2
                ) if unclassified_transactions else 0
            }
        }
    
    def get_improvement_suggestions(self, analysis_result: Dict) -> List[str]:
        """
        분석 결과를 바탕으로 개선 제안사항 생성
        
        Args:
            analysis_result: analyze_unclassified_batch 결과
            
        Returns:
            List[str]: 개선 제안사항 목록
        """
        suggestions = []
        
        # 1. 상위 키워드 기반 제안
        top_keywords = analysis_result.get("top_keywords", [])
        if top_keywords:
            most_common = top_keywords[0][0]
            suggestions.append(f"💡 '{most_common}' 키워드가 가장 많이 나타납니다. 관련 분류 규칙을 추가하세요.")
        
        # 2. 재분류 가능한 항목 제안
        recategorizable = analysis_result["summary"].get("suggested_recategorization", 0)
        if recategorizable > 0:
            suggestions.append(f"📊 {recategorizable}개 카테고리로 재분류 가능한 거래가 있습니다.")
        
        # 3. 금액 기반 패턴 제안
        avg_amount = analysis_result["summary"].get("avg_transaction_amount", 0)
        if avg_amount > 100000:
            suggestions.append("💰 고액 거래가 많습니다. 주거비나 대형 쇼핑 패턴을 확인해보세요.")
        elif avg_amount < 10000:
            suggestions.append("💳 소액 거래가 많습니다. 편의점이나 카페 패턴을 확인해보세요.")
        
        # 4. 카테고리별 신뢰도 기반 제안
        category_suggestions = analysis_result.get("category_suggestions", {})
        high_confidence_categories = [
            cat for cat, data in category_suggestions.items() 
            if data["avg_confidence"] > 80
        ]
        
        if high_confidence_categories:
            suggestions.append(f"✅ {', '.join(high_confidence_categories)} 카테고리는 높은 신뢰도로 재분류 가능합니다.")
        
        return suggestions

# 전역 인스턴스
business_search_service = BusinessSearchService()