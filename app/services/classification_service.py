# 파일 위치: app/services/classification_service.py
# 설명: 송금을 별도 분류하는 고급 AI 거래내역 분류 서비스
# 협업 가이드: 
# 1. 송금 감지 후 별도 분류 (사람 이름 + 송금 서비스)
# 2. 송금이 아닌 경우에만 기존 AI 분류 로직 실행
# 3. 프론트엔드에서 "송금" 카테고리는 소비 비율 계산에서 제외
# 4. 송금 정보는 별도 영역에 표시

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import re
import random
import numpy as np
from difflib import get_close_matches
import logging

# 선택적 import (설치되지 않은 경우 기본 모드로 동작)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
    print("✅ BERT 모델 사용 가능")
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT 모델 미설치, 기본 모드로 동작")

try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
    print("✅ KoNLPy 형태소 분석 사용 가능")
except ImportError:
    KONLPY_AVAILABLE = False
    print("⚠️ KoNLPy 미설치, 형태소 분석 비활성화")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from scipy import sparse
    from scipy.sparse import csr_matrix
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn 사용 가능")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn 미설치, 기본 분류만 사용")

class TransactionClassificationService:
    """
    송금을 별도 분류하는 고급 AI 거래내역 분류 서비스
    (송금 분리 + 우수한 분류 성능 + 프로덕션 안정성)
    """
    def __init__(self):
        # 🔥 송금 감지 패턴 정의
        self.REMITTANCE_PATTERNS = {
            "person_name_patterns": [
                r'^[가-힣]{2,3}$',  # 2-3글자 한글 이름
                r'^[가-힣]{2,4}님$',  # 이름 + 님
                r'^[가-힣]{2,4}\s*씨$',  # 이름 + 씨
            ],
            "remittance_services": [
                "토스페이", "카카오페이", "페이코", "네이버페이", "삼성페이",
                "송금", "이체", "입금", "용돈", "계좌이체", "무통장입금",
                "온라인이체", "ATM이체", "모바일송금", "간편송금"
            ],
            "exclude_keywords": [
                # 송금으로 오인될 수 있지만 실제로는 일반 소비인 것들
                "마트", "편의점", "카페", "음식점", "병원", "약국",
                "주유소", "대학교", "학원", "회사", "상점", "매장", "점"
            ]
        }
        
        # 카테고리 정의 (송금 카테고리 추가)
        self.CATEGORIES: Dict[str, List[str]] = {
            "주거비":        ["한국전력공사","서울도시가스","한국수력원자력","LH토지주택공사","서울주택도시공사","예스코","한전"],
            "교통비":        ["카카오택시","T맵택시","고속버스","SRT","코레일","지하철"],
            "통신비":        ["SKT","KT","LGU+","알뜰폰샵","데이터충전","LGU+인터넷"],
            "의료비":        ["서울대병원","삼성서울병원","아산병원","강남연세치과","메디힐피부과"],
            "교육비":        ["YBM어학원","메가스터디","해커스어학원","에듀윌","해커스패스","프린트","CHATGPT"],
            "식비":          ["맥도날드","스타벅스","이디야커피","배달의민족","요기요","컴포즈커피","어오네","옹기종기","빽다방","반점"],
            "생활용품비":    ["GS25","CU","이마트","롯데마트","홈플러스","세븐일레븐"],
            "이미용/의류/화장품": ["무신사","H&M","ZARA","아모레퍼시픽","에뛰드하우스","TEMU"],
            "온라인 컨텐츠":  ["넷플릭스","왓챠","유튜브프리미엄","멜론","쿠팡플레이","SOOP"],
            "여가비":        ["CGV","롯데시네마","서울랜드","롯데월드","스타필드","노래","PC"],
            "기타":          ["11번가","쿠팡","옥션","G마켓","인터파크"],
            "송금":          []  # 🔥 새로 추가된 송금 카테고리
        }

        # 룰 기반 키워드 (우선순위)
        self.RULES: List[tuple] = [
            ("통신비",   ["통신","인터넷","LGU+","SKT","KT"]),
            ("식비",     ["배민","요기요","맥도날드","스타벅스","커피","이디야"]),
            ("교통비",   ["택시","버스","지하철","SRT","코레일"]),
            ("의료비",   ["병원","치과","약국","피부과"]),
        ]

        # 퍼지 매칭용 샘플
        self.CATEGORY_MAP: Dict[str, List[str]] = {cat: samples[:] for cat, samples in self.CATEGORIES.items()}
        self.ALL_SAMPLES: List[str] = [s for samples in self.CATEGORY_MAP.values() for s in samples if samples]  # 빈 리스트 제외
        
        # 파라미터
        self.DESIRED_PER_CAT = 1000
        self.AUG_FACTOR = 1
        self.TFIDF_DIM = 50
        self.MORPH_DIM = 5
        self.BATCH_SIZE = 128
        self.RANDOM_STATE = 42
        self.RSEARCH_ITERS = 4
        self.RSEARCH_CV = 2
        
        # 초기화 상태
        self.model_initialized = False
        self.ai_model_available = False
        
        # 모델 변수들 (None으로 초기화)
        self.tokenizer = None
        self.bert_model = None
        self.tfidf_vectorizer = None
        self.svd_tfidf = None
        self.count_vectorizer = None
        self.svd_morph = None
        self.okt = None
        self.label_encoder = None
        self.ml_classifier = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🚀 송금 분리 거래내역 분류 서비스 초기화 완료")

    def _is_remittance(self, name: str) -> bool:
        """
        송금 여부 판단 함수
        
        Args:
            name: 정규화된 거래처명
            
        Returns:
            bool: 송금이면 True, 아니면 False
        """
        if not name or not name.strip():
            return False
            
        name_lower = name.lower().strip()
        
        # 1. 제외 키워드 체크 (일반 소비로 확실한 것들)
        for exclude_keyword in self.REMITTANCE_PATTERNS["exclude_keywords"]:
            if exclude_keyword in name_lower:
                return False
        
        # 2. 사람 이름 패턴 체크
        for pattern in self.REMITTANCE_PATTERNS["person_name_patterns"]:
            if re.match(pattern, name):
                return True
        
        # 3. 송금 서비스 키워드 체크
        for service in self.REMITTANCE_PATTERNS["remittance_services"]:
            if service.lower() in name_lower:
                return True
                
        return False

    def initialize_model(self):
        """지연 초기화 + 랜덤 시드 고정"""
        if self.model_initialized:
            return
            
        try:
            # 랜덤 시드 고정
            random.seed(self.RANDOM_STATE)
            np.random.seed(self.RANDOM_STATE)
            
            print("🤖 모델 초기화 중...")
            self.model_initialized = True
            print("✅ 기본 분류 완료")
            
            # AI 모델들이 모두 사용 가능한 경우에만 AI 모드 활성화
            if BERT_AVAILABLE and SKLEARN_AVAILABLE:
                print("🔥 AI 모델 로딩 및 학습 시작")
                self._initialize_ai_models()
                self.ai_model_available = True
                print("✅ AI 모델 준비 완료")
            else:
                print("⚠️ AI 패키지 미설치, 기본 모드로 동작")
                
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            print("⚠️ AI 모델 초기화 실패, 기본 모드로 동작")

    def _initialize_ai_models(self):
        """AI 모델들 초기화"""
        try:
            if BERT_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
                self.bert_model = AutoModel.from_pretrained("monologg/koelectra-small-v3-discriminator").eval()
                
            if KONLPY_AVAILABLE:
                self.okt = Okt()
                
            self._train_ai_models()
            
        except Exception as e:
            self.logger.error(f"AI 모델 초기화 실패: {e}")
            self.ai_model_available = False
            raise

    def _train_ai_models(self):
        """AI 모델 학습"""
        try:
            # 학습 데이터 생성 (송금 카테고리 제외)
            training_data = self._generate_training_data()
            docs, labels = zip(*training_data)
            
            # 피처 추출기 학습
            self._train_feature_extractors(docs, labels)
            
            # ML 모델 학습
            self._train_ml_model(docs, labels)
            
        except Exception as e:
            self.logger.error(f"AI 모델 학습 실패: {e}")
            self.ai_model_available = False
            raise

    def _generate_training_data(self):
        """학습 데이터 생성 (증강 포함) - 송금 카테고리 제외"""
        try:
            # 기본 데이터 생성 (송금 카테고리는 제외)
            base = []
            for cat, samples in self.CATEGORIES.items():
                if cat == "송금" or not samples:  # 송금 카테고리와 빈 샘플은 제외
                    continue
                    
                for i in range(self.DESIRED_PER_CAT):
                    base_sample = samples[i % len(samples)]
                    suffix = i // len(samples) + 1
                    name = f"{base_sample}{suffix:03d}" if suffix > 1 else base_sample
                    base.append((name, cat))

            # 데이터 증강
            augmented = []
            for name, cat in base:
                normalized = self.normalize(name)
                augmented.append((normalized, cat))
                
                # 추가 증강
                for _ in range(self.AUG_FACTOR):
                    augmented_name = self._augment_name(name)
                    augmented.append((self.normalize(augmented_name), cat))
                    
            return augmented
            
        except Exception as e:
            self.logger.error(f"학습 데이터 생성 실패: {e}")
            raise

    def _augment_name(self, name: str) -> str:
        """데이터 증강"""
        try:
            # 공백 삽입
            if random.random() < 0.3 and len(name) > 1:
                pos = random.randint(1, len(name) - 1)
                name = name[:pos] + " " + name[pos:]
                
            # 언더스코어 변환
            if random.random() < 0.2 and " " in name:
                name = name.replace(" ", "_", 1)
                
            # "점" 추가
            if random.random() < 0.2:
                name = name + "점"
                
            # 별표 변환
            if random.random() < 0.1 and " " in name:
                name = name.replace(" ", "*", 1)
                
            return name
            
        except Exception as e:
            self.logger.warning(f"데이터 증강 중 오류: {e}")
            return name

    def _train_feature_extractors(self, docs: List[str], labels: List[str]):
        """피처 추출기들 학습"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # TF-IDF + SVD
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer="char", 
                ngram_range=(2, 4), 
                min_df=5
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(docs)
            self.svd_tfidf = TruncatedSVD(
                n_components=self.TFIDF_DIM, 
                random_state=self.RANDOM_STATE
            )
            self.svd_tfidf.fit(X_tfidf)
            
            # 형태소 분석 + SVD (KoNLPy 사용 가능한 경우)
            if KONLPY_AVAILABLE and self.okt:
                tokens = [" ".join(self.okt.nouns(doc)) for doc in docs]
                self.count_vectorizer = CountVectorizer(min_df=5)
                X_morph = self.count_vectorizer.fit_transform(tokens)
                self.svd_morph = TruncatedSVD(
                    n_components=self.MORPH_DIM, 
                    random_state=self.RANDOM_STATE
                )
                self.svd_morph.fit(X_morph)
                
        except Exception as e:
            self.logger.error(f"피처 추출기 학습 실패: {e}")
            raise

    def _train_ml_model(self, docs: List[str], labels: List[str]):
        """ML 모델 학습"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # 피처 추출
            X = self._extract_features_batch(docs)
            
            # 라벨 인코딩
            self.label_encoder = LabelEncoder().fit(labels)
            y = self.label_encoder.transform(labels)
            
            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.RANDOM_STATE
            )
            
            # 하이퍼파라미터 탐색
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.RANDOM_STATE),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10],
                    "min_samples_leaf": [1, 2]
                },
                n_iter=self.RSEARCH_ITERS,
                cv=self.RSEARCH_CV,
                n_jobs=-1,
                random_state=self.RANDOM_STATE
            )
            
            search.fit(X_train, y_train)
            self.ml_classifier = search.best_estimator_
            
            # 성능 평가
            y_pred = self.ml_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"🎯 모델 정확도: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"ML 모델 학습 실패: {e}")
            self.ai_model_available = False
            raise

    def _extract_features_batch(self, texts: List[str]):
        """배치 피처 추출"""
        try:
            features = []
            
            # BERT 피처
            if BERT_AVAILABLE and self.bert_model and self.tokenizer:
                bert_features = self._batch_bert(texts)
                features.append(csr_matrix(bert_features))
            
            # TF-IDF 피처
            if self.tfidf_vectorizer and self.svd_tfidf:
                tfidf_features = self.tfidf_vectorizer.transform(texts)
                tfidf_reduced = self.svd_tfidf.transform(tfidf_features)
                features.append(csr_matrix(tfidf_reduced))
            
            # 형태소 피처
            if KONLPY_AVAILABLE and self.okt and self.count_vectorizer and self.svd_morph:
                tokens = [" ".join(self.okt.nouns(text)) for text in texts]
                morph_features = self.count_vectorizer.transform(tokens)
                morph_reduced = self.svd_morph.transform(morph_features)
                features.append(csr_matrix(morph_reduced))
            
            # 수치형 피처
            lengths = np.array([len(text) for text in texts])[:, None]
            digit_counts = np.array([sum(c.isdigit() for c in text) for text in texts])[:, None]
            numeric_features = np.hstack([lengths, digit_counts])
            features.append(csr_matrix(numeric_features))
            
            # 모든 피처 결합
            if features:
                return sparse.hstack(features)
            else:
                # 피처가 없는 경우 기본 수치형만 반환
                return csr_matrix(numeric_features)
                
        except Exception as e:
            self.logger.error(f"피처 추출 실패: {e}")
            # 최소한의 피처라도 반환
            lengths = np.array([len(text) for text in texts])[:, None]
            return csr_matrix(lengths)

    def _batch_bert(self, texts: List[str]):
        """배치 BERT 인코딩"""
        try:
            vectors = []
            for i in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[i:i + self.BATCH_SIZE]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=32
                )
                
                with torch.no_grad():
                    outputs = self.bert_model(**encoded)
                    
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
                
            return np.vstack(vectors)
            
        except Exception as e:
            self.logger.error(f"BERT 인코딩 실패: {e}")
            # 빈 벡터 반환
            return np.zeros((len(texts), 768))

    def normalize(self, name: str) -> str:
        """텍스트 정규화"""
        if not name or not isinstance(name, str):
            return ""
            
        try:
            normalized = name.lower().replace("_", " ").replace("*", " ")
            normalized = re.sub(r"(점$|\d{3}$)", "", normalized)
            return normalized.strip()
            
        except Exception as e:
            self.logger.warning(f"정규화 실패: {e}")
            return str(name).strip()

    def rule_based_category(self, name: str) -> Optional[str]:
        """룰 기반 분류"""
        try:
            for category, keywords in self.RULES:
                for keyword in keywords:
                    if keyword.lower() in name.lower():
                        return category
            return None
            
        except Exception as e:
            self.logger.warning(f"룰 기반 분류 실패: {e}")
            return None

    def fuzzy_category(self, name: str, threshold: int = 90) -> Optional[str]:
        """퍼지 매칭 분류"""
        try:
            matches = get_close_matches(
                name, 
                self.ALL_SAMPLES, 
                n=1, 
                cutoff=threshold/100
            )
            
            if matches:
                match = matches[0]
                for category, samples in self.CATEGORY_MAP.items():
                    if match in samples:
                        return category
                        
            return None
            
        except Exception as e:
            self.logger.warning(f"퍼지 매칭 실패: {e}")
            return None

    def categorize_store(self, name: str) -> str:
        """
        🔥 송금 분리 + 향상된 분류 함수
        순서: 송금 체크 → 룰 기반 → 퍼지 매칭 → AI 모델
        """
        # 모델 초기화 확인
        if not self.model_initialized:
            self.initialize_model()
            
        # 입력 검증
        if not name or not name.strip():
            return "기타"
            
        try:
            normalized = self.normalize(name)
            
            # 🔥 1단계: 송금 여부 먼저 체크
            if self._is_remittance(normalized):
                return "송금"

            # 2단계: 룰 기반 분류
            rule_result = self.rule_based_category(normalized)
            if rule_result:
                return rule_result

            # 3단계: 퍼지 매칭
            fuzzy_result = self.fuzzy_category(normalized, threshold=90)
            if fuzzy_result:
                return fuzzy_result

            # 4단계: AI 모델 분류 (사용 가능한 경우)
            if self.ai_model_available and self.ml_classifier and self.label_encoder:
                try:
                    features = self._extract_features_batch([normalized])
                    prediction = self.ml_classifier.predict(features)[0]
                    category = self.label_encoder.inverse_transform([prediction])[0]
                    return category
                    
                except Exception as e:
                    self.logger.warning(f"AI 모델 예측 실패, 폴백 분류로 전환: {e}")

            # 5단계: 최종 폴백: 기본 키워드 매칭
            for category, samples in self.CATEGORIES.items():
                if category == "송금":  # 송금은 이미 체크했으므로 제외
                    continue
                for sample in samples:
                    if sample.lower() in normalized or normalized in sample.lower():
                        return category

            # 6단계: 모든 방법 실패 시
            return "기타"
            
        except Exception as e:
            self.logger.error(f"분류 중 예외 발생: {e}")
            return "기타"

    def parse_nh_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """NH농협 엑셀 파일 파싱 (안정성 강화)"""
        if not self.model_initialized:
            self.initialize_model()
            
        try:
            # 파일 읽기
            raw_df = pd.read_excel(file_path, header=None)
            
            # 헤더 찾기
            header_rows = raw_df[raw_df.apply(
                lambda row: row.astype(str).str.contains("순번").any(), axis=1
            )].index
            
            if len(header_rows) == 0:
                raise ValueError("'순번' 헤더를 찾을 수 없습니다.")
                
            # 실제 데이터 읽기
            df = pd.read_excel(file_path, header=header_rows[0])
            
            # 필수 컬럼 찾기
            required_columns = {
                '거래기록사항': None,
                '출금금액': None,
                '거래일시': None
            }
            
            for col in df.columns:
                col_str = str(col)
                for req_col in required_columns:
                    if req_col in col_str and required_columns[req_col] is None:
                        required_columns[req_col] = col
                        
            # 누락된 컬럼 확인
            missing_columns = [k for k, v in required_columns.items() if v is None]
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {missing_columns}")
            
            # 데이터 전처리
            df = df[df[required_columns['출금금액']].notna()]
            df = df[df[required_columns['출금금액']].astype(str).str.strip() != ""]
            df = df[df[required_columns['출금금액']].astype(str).str.strip() != "0"]
            
            # 분류 수행
            store_names = df[required_columns['거래기록사항']].astype(str).str.strip().tolist()
            categories = [self.categorize_store(name) for name in store_names]
            
            # 결과 구성
            result = []
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    # 금액 파싱
                    amount_str = str(row[required_columns['출금금액']]).strip().replace(",", "")
                    amount = float(re.sub(r"[^\d.]", "", amount_str))
                    
                    if amount <= 0:
                        continue
                        
                    result.append({
                        "transaction_date": row.get(required_columns['거래일시'], datetime.now()),
                        "amount": amount,
                        "store_name": store_names[i],
                        "category": categories[i],
                        "description": store_names[i],
                        "payment_method": "카드"
                    })
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"행 {i} 처리 중 스킵: {e}")
                    continue
                    
            return result
            
        except Exception as e:
            self.logger.error(f"NH 엑셀 파싱 실패: {e}")
            raise

    def parse_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """CSV 파일 파싱 (안정성 강화)"""
        if not self.model_initialized:
            self.initialize_model()
            
        try:
            # 다양한 인코딩으로 시도
            df = None
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError("지원되는 인코딩을 찾을 수 없습니다.")
            
            # 컬럼 자동 매핑
            column_mapping = {
                'date': ['날짜', '거래일', '일자', 'date', '거래일시'],
                'amount': ['금액', '출금액', '입금액', '사용금액', 'amount', '출금금액'],
                'merchant': ['가맹점', '상호명', '적요', 'merchant', '거래기록사항']
            }
            
            found_columns = {}
            for target, candidates in column_mapping.items():
                for col in df.columns:
                    col_lower = str(col).lower()
                    for candidate in candidates:
                        if candidate.lower() in col_lower and target not in found_columns:
                            found_columns[target] = col
                            break
                    if target in found_columns:
                        break
            
            # 필수 컬럼 확인
            if 'amount' not in found_columns or 'merchant' not in found_columns:
                raise ValueError("필수 컬럼(금액, 가맹점)을 찾을 수 없습니다.")
            
            # 데이터 처리
            result = []
            valid_rows = []
            
            for idx, row in df.iterrows():
                try:
                    # 금액 처리
                    amount_str = str(row[found_columns['amount']]).replace(',', '').strip()
                    if not amount_str or amount_str == 'nan':
                        continue
                        
                    amount = float(re.sub(r"[^\d.]", "", amount_str))
                    if amount <= 0:
                        continue
                    
                    # 가맹점명 처리
                    merchant_name = str(row[found_columns['merchant']]).strip()
                    if not merchant_name or merchant_name == 'nan':
                        merchant_name = '기타'
                    
                    valid_rows.append((idx, merchant_name, amount, row))
                    
                except (ValueError, TypeError):
                    continue
            
            # 배치 분류
            merchant_names = [name for _, name, _, _ in valid_rows]
            categories = [self.categorize_store(name) for name in merchant_names]
            
            # 결과 구성
            for i, (idx, name, amount, row) in enumerate(valid_rows):
                transaction_date = datetime.now()
                if 'date' in found_columns:
                    try:
                        transaction_date = pd.to_datetime(row[found_columns['date']])
                    except:
                        pass
                
                result.append({
                    'transaction_date': transaction_date,
                    'amount': amount,
                    'store_name': name,
                    'category': categories[i],
                    'description': name,
                    'payment_method': '카드'
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"CSV 파싱 실패: {e}")
            raise

    def get_category_statistics(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        🔥 송금 분리 통계 생성
        
        Args:
            transactions: 거래 내역 리스트
            
        Returns:
            Dict: 송금 분리된 통계 정보
        """
        try:
            if not transactions:
                return {
                    "consumption_analysis": {},
                    "remittance_info": {},
                    "total_transactions": 0
                }
            
            # 송금과 실제 소비 분리
            remittance_transactions = [t for t in transactions if t.get('category') == '송금']
            consumption_transactions = [t for t in transactions if t.get('category') != '송금']
            
            # 실제 소비 카테고리별 집계
            consumption_by_category = {}
            total_consumption = 0
            
            for transaction in consumption_transactions:
                category = transaction.get('category', '기타')
                amount = float(transaction.get('amount', 0))
                
                if category not in consumption_by_category:
                    consumption_by_category[category] = 0
                consumption_by_category[category] += amount
                total_consumption += amount
            
            # 실제 소비 비율 계산 (송금 제외)
            consumption_percentages = {}
            if total_consumption > 0:
                for category, amount in consumption_by_category.items():
                    consumption_percentages[category] = round((amount / total_consumption) * 100, 1)
            
            # 송금 정보 집계
            total_remittance = sum(float(t.get('amount', 0)) for t in remittance_transactions)
            remittance_count = len(remittance_transactions)
            
            # 전체 거래 대비 송금 비율
            total_amount = sum(float(t.get('amount', 0)) for t in transactions)
            remittance_percentage = round((total_remittance / total_amount) * 100, 1) if total_amount > 0 else 0
            
            return {
                "consumption_analysis": {
                    "categories": consumption_by_category,
                    "percentages": consumption_percentages,
                    "total_amount": total_consumption,
                    "transaction_count": len(consumption_transactions)
                },
                "remittance_info": {
                    "total_amount": total_remittance,
                    "transaction_count": remittance_count,
                    "percentage_of_total": remittance_percentage,
                    "transactions": remittance_transactions[:10]  # 최근 10건만
                },
                "total_transactions": len(transactions),
                "summary": {
                    "consumption_total": total_consumption,
                    "remittance_total": total_remittance,
                    "grand_total": total_amount
                }
            }
            
        except Exception as e:
            self.logger.error(f"통계 생성 실패: {e}")
            return {
                "consumption_analysis": {},
                "remittance_info": {},
                "total_transactions": 0,
                "error": str(e)
            }

    def analyze_file_and_classify(self, file_path: str, file_type: str = "auto") -> Dict[str, Any]:
        """
        파일 분석 및 분류 통합 함수
        
        Args:
            file_path: 파일 경로
            file_type: 파일 타입 ("excel", "csv", "auto")
            
        Returns:
            Dict: 분류 결과 및 통계
        """
        try:
            # 파일 타입 자동 감지
            if file_type == "auto":
                if file_path.lower().endswith(('.xlsx', '.xls')):
                    file_type = "excel"
                elif file_path.lower().endswith('.csv'):
                    file_type = "csv"
                else:
                    raise ValueError("지원하지 않는 파일 형식입니다.")
            
            # 파일 파싱
            if file_type == "excel":
                transactions = self.parse_nh_excel(file_path)
            elif file_type == "csv":
                transactions = self.parse_csv_file(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
            
            # 통계 생성
            statistics = self.get_category_statistics(transactions)
            
            return {
                "success": True,
                "transactions": transactions,
                "statistics": statistics,
                "message": f"총 {len(transactions)}건의 거래를 분류했습니다."
            }
            
        except Exception as e:
            self.logger.error(f"파일 분석 실패: {e}")
            return {
                "success": False,
                "transactions": [],
                "statistics": {},
                "error": str(e),
                "message": "파일 분석 중 오류가 발생했습니다."
            }

# 전역 인스턴스 생성
classification_service = TransactionClassificationService()