# 파일 위치: app/services/classification_service.py
# 설명: 고급 AI 기법이 통합된 거래내역 분류 서비스 (기존 기능 완전 보존)

from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import re
import tempfile
import os
import random
import numpy as np
from difflib import get_close_matches

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
    from sklearn.metrics import classification_report, accuracy_score
    from scipy import sparse
    from scipy.sparse import csr_matrix
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn 사용 가능")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn 미설치, 기본 분류만 사용")


class TransactionClassificationService:
    """
    고급 AI 기법이 통합된 거래내역 분류 서비스
    
    기존의 모든 기능을 유지하면서 고급 AI 분류 기법을 추가한 버전
    - 기존: 룰 기반 + 샘플 매칭 분류
    - 추가: BERT + TF-IDF + 형태소 분석 + 머신러닝 분류
    """
    
    def __init__(self):
        # ========== 기존 카테고리 정의 (완전 보존) ==========
        self.CATEGORIES = {
            "주거비":        ["한국전력공사","서울도시가스","한국수력원자력","LH토지주택공사","서울주택도시공사","예스코","한전"],
            "교통비":        ["카카오택시","T맵택시","고속버스","SRT","코레일","지하철"],
            "통신비":        ["SKT","KT","LGU+","알뜰폰샵","데이터충전","LGU+인터넷"],
            "의료비":        ["서울대병원","삼성서울병원","아산병원","강남연세치과","메디힐피부과"],
            "교육비":        ["YBM어학원","메가스터디","해커스어학원","에듀윌","해커스패스","프린트","CHATGPT"],
            "식비":          ["맥도날드","스타벅스","이디야커피","배달의민족","요기요","컴포즈커피","어오네","옹기종기","빽다방"],
            "생활용품비":    ["GS25","CU","이마트","롯데마트","홈플러스","세븐일레븐"],
            "이미용/의류/화장품": ["무신사","H&M","ZARA","아모레퍼시픽","에뛰드하우스","TEMU"],
            "온라인 컨텐츠":  ["넷플릭스","왓챠","유튜브프리미엄","멜론","쿠팡플레이","SOOP"],
            "여가비":        ["CGV","롯데시네마","서울랜드","롯데월드","스타필드","노래","PC"],
            "기타":          ["11번가","쿠팡","옥션","G마켓","인터파크","토스페이","카카오페이"]
        }

        
        # ========== 기존 룰 기반 키워드 (완전 보존) ==========
        self.RULES = [
            ("통신비", ["통신", "인터넷", "LGU+", "SKT", "KT", "휴대폰"]),
            ("식비", ["배민", "요기요", "맥도날드", "스타벅스", "커피", "이디야", "치킨", "피자"]),
            ("교통비", ["택시", "버스", "지하철", "SRT", "코레일", "고속버스"]),
            ("의료비", ["병원", "치과", "약국", "피부과", "한의원"]),
            ("주거비", ["전기", "가스", "수도", "관리비", "월세"]),
            ("생활용품비", ["편의점", "마트", "이마트", "롯데마트", "홈플러스"])
        ]
        
        # ========== 고급 AI 기법 추가 설정 ==========
        # 퍼지 매칭용 데이터
        self.CATEGORY_MAP = {cat: samples[:] for cat, samples in self.CATEGORIES.items()}
        self.ALL_SAMPLES = [s for samples in self.CATEGORY_MAP.values() for s in samples]
        
        # AI 모델 관련 설정
        self.DESIRED_PER_CAT = 1000
        self.AUG_FACTOR = 1
        self.TFIDF_DIM = 50
        self.MORPH_DIM = 5
        self.BATCH_SIZE = 128
        self.RANDOM_STATE = 42
        self.RSEARCH_ITERS = 4
        self.RSEARCH_CV = 2
        
        # 모델 초기화 상태
        self.model_initialized = False
        self.ai_model_available = False
        
        # AI 모델 관련 변수들
        self.tokenizer = None
        self.bert_model = None
        self.tfidf_vectorizer = None
        self.svd_tfidf = None
        self.count_vectorizer = None
        self.svd_morph = None
        self.okt = None
        self.label_encoder = None
        self.ml_classifier = None
        
        print("🚀 고급 AI 거래내역 분류 서비스 초기화 완료")
    
    def initialize_model(self):
        """모델 초기화 (기존 기능 + 고급 AI 기능)"""
        if self.model_initialized:
            return
            
        print("🤖 분류 모델 초기화 중...")
        
        # 기존 간소화 모델은 항상 사용 가능
        self.model_initialized = True
        print("✅ 기본 분류 모델 초기화 완료")
        
        # 고급 AI 모델 초기화 시도
        try:
            if BERT_AVAILABLE and SKLEARN_AVAILABLE:
                print("🔥 고급 AI 모델 초기화 시작...")
                self._initialize_ai_models()
                self.ai_model_available = True
                print("✅ 고급 AI 모델 초기화 완료!")
            else:
                print("⚠️ 고급 AI 패키지 미설치 - 기본 모드로 동작")
                
        except Exception as e:
            print(f"❌ 고급 AI 모델 초기화 실패: {e}")
            print("⚠️ 기본 룰 기반 모드로 동작합니다.")
            self.ai_model_available = False
    
    def _initialize_ai_models(self):
        """고급 AI 모델들 초기화"""
        # 1. BERT 모델 초기화
        if BERT_AVAILABLE:
            print("🔥 BERT 모델 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
            self.bert_model = AutoModel.from_pretrained("monologg/koelectra-small-v3-discriminator").eval()
            print("✅ BERT 모델 로딩 완료")
        
        # 2. 형태소 분석기 초기화
        if KONLPY_AVAILABLE:
            print("🔥 형태소 분석기 초기화 중...")
            self.okt = Okt()
            print("✅ 형태소 분석기 초기화 완료")
        
        # 3. 훈련 데이터 생성 및 모델 훈련
        print("🔥 훈련 데이터 생성 및 모델 훈련 중...")
        self._train_ai_models()
        print("✅ AI 모델 훈련 완료")
    
    def _train_ai_models(self):
        """AI 모델 훈련"""
        try:
            # 훈련 데이터 생성
            training_data = self._generate_training_data()
            docs = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]
            
            # 피처 추출기 훈련
            self._train_feature_extractors(docs, labels)
            
            # 머신러닝 모델 훈련
            self._train_ml_model(docs, labels)
            
        except Exception as e:
            print(f"❌ AI 모델 훈련 실패: {e}")
            self.ai_model_available = False
    
    def _generate_training_data(self):
        """훈련 데이터 생성"""
        base = []
        for cat, samples in self.CATEGORIES.items():
            for i in range(self.DESIRED_PER_CAT):
                b = samples[i % len(samples)]
                suf = i // len(samples) + 1
                nm = f"{b}{suf:03d}" if suf > 1 else b
                base.append((nm, cat))
        
        # 데이터 증강
        aug = []
        for nm, cat in base:
            n0 = self.normalize(nm)
            aug.append((n0, cat))
            for _ in range(self.AUG_FACTOR):
                aug.append((self.normalize(self._augment_name(nm)), cat))
        
        return aug
    
    def _augment_name(self, name: str) -> str:
        """이름 증강"""
        if random.random() < 0.3 and len(name) > 1:
            p = random.randint(1, len(name) - 1)
            name = name[:p] + " " + name[p:]
        if random.random() < 0.2 and " " in name:
            name = name.replace(" ", "_", 1)
        if random.random() < 0.2:
            name = name + "점"
        if random.random() < 0.1 and " " in name:
            name = name.replace(" ", "*", 1)
        return name
    
    def _train_feature_extractors(self, docs, labels):
        """피처 추출기 훈련"""
        if not SKLEARN_AVAILABLE:
            return
            
        # TF-IDF + SVD
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=5)
        X_tfidf = self.tfidf_vectorizer.fit_transform(docs)
        self.svd_tfidf = TruncatedSVD(n_components=self.TFIDF_DIM, random_state=self.RANDOM_STATE)
        self.svd_tfidf.fit(X_tfidf)
        
        # 형태소 분석 + SVD (KoNLPy 사용 가능한 경우)
        if KONLPY_AVAILABLE and self.okt:
            tokens = [" ".join(self.okt.nouns(s)) for s in docs]
            self.count_vectorizer = CountVectorizer(min_df=5)
            X_morph = self.count_vectorizer.fit_transform(tokens)
            self.svd_morph = TruncatedSVD(n_components=self.MORPH_DIM, random_state=self.RANDOM_STATE)
            self.svd_morph.fit(X_morph)
    
    def _train_ml_model(self, docs, labels):
        """머신러닝 모델 훈련"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # 피처 추출
            X_all = self._extract_features_batch(docs)
            
            # 라벨 인코딩
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            
            # 훈련/테스트 분할
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y, test_size=0.2, stratify=y, random_state=self.RANDOM_STATE
            )
            
            # 하이퍼파라미터 튜닝
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.RANDOM_STATE),
                {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_leaf": [1, 2]},
                n_iter=self.RSEARCH_ITERS, cv=self.RSEARCH_CV, n_jobs=-1, verbose=0, random_state=self.RANDOM_STATE
            )
            search.fit(X_tr, y_tr)
            self.ml_classifier = search.best_estimator_
            
            # 성능 평가
            y_pred = self.ml_classifier.predict(X_te)
            accuracy = accuracy_score(y_te, y_pred)
            print(f"🎯 AI 모델 정확도: {accuracy:.3f}")
            
        except Exception as e:
            print(f"❌ ML 모델 훈련 실패: {e}")
            self.ml_classifier = None
    
    def _extract_features_batch(self, texts):
        """배치 피처 추출"""
        features = []
        
        # BERT 피처 (사용 가능한 경우)
        if BERT_AVAILABLE and self.tokenizer and self.bert_model:
            bert_features = self._batch_bert(texts)
            features.append(sparse.csr_matrix(bert_features))
        
        # TF-IDF 피처
        if self.tfidf_vectorizer and self.svd_tfidf:
            tfidf_features = self.svd_tfidf.transform(self.tfidf_vectorizer.transform(texts))
            features.append(sparse.csr_matrix(tfidf_features))
        
        # 형태소 피처 (사용 가능한 경우)
        if KONLPY_AVAILABLE and self.okt and self.count_vectorizer and self.svd_morph:
            tokens = [" ".join(self.okt.nouns(s)) for s in texts]
            morph_features = self.svd_morph.transform(self.count_vectorizer.transform(tokens))
            features.append(sparse.csr_matrix(morph_features))
        
        # 수치형 피처
        lengths = np.array([len(s) for s in texts])[:, None]
        digit_counts = np.array([sum(c.isdigit() for c in s) for s in texts])[:, None]
        num_features = np.hstack([lengths, digit_counts])
        features.append(sparse.csr_matrix(num_features))
        
        if features:
            return sparse.hstack(features)
        else:
            # 기본 피처 (길이만)
            return sparse.csr_matrix(lengths)
    
    def _batch_bert(self, texts):
        """배치 BERT 임베딩"""
        vectors = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
            with torch.no_grad():
                output = self.bert_model(**encoded)
            vectors.append(output.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(vectors)
    
    def normalize(self, name: str) -> str:
        """상호명 정규화 (기존 기능 완전 보존)"""
        if not isinstance(name, str):
            name = str(name)
        
        # 소문자 변환 및 특수문자 제거
        n = name.lower().replace("_", " ").replace("*", " ")
        # 지점명, 숫자 제거
        n = re.sub(r"(점$|\d{3}$)", "", n)
        return n.strip()
    
    def rule_based_category(self, name: str) -> str:
        """룰 기반 카테고리 분류 (기존 기능 완전 보존)"""
        for cat, kws in self.RULES:
            for kw in kws:
                if kw.lower() in name.lower():
                    return cat
        return None
    
    def fuzzy_category(self, name: str, threshold: int = 90) -> str:
        """퍼지 매칭 카테고리 분류 (고급 기능 추가)"""
        matches = get_close_matches(name, self.ALL_SAMPLES, n=1, cutoff=threshold/100)
        if matches:
            match = matches[0]
            for cat, samples in self.CATEGORY_MAP.items():
                if match in samples:
                    return cat
        return None
    
    def categorize_store(self, name: str) -> str:
        """
        상호명 기반 카테고리 분류 (기존 + 고급 AI 통합)
        
        분류 우선순위:
        1. 기존 룰 기반 분류 (최우선)
        2. 퍼지 매칭 (고급 기능)
        3. AI 머신러닝 분류 (최고급 기능)
        4. 기존 샘플 기반 분류 (폴백)
        """
        if not name or str(name).strip() == '':
            return "기타"
            
        n = self.normalize(str(name))
        
        # 0) 사람 이름 감지 (한글 2-3글자)
        if re.fullmatch(r'[가-힣]{2,3}', n):
            return "기타"
        
        # 1) 기존 룰 기반 분류 (최우선)
        rule_result = self.rule_based_category(n)
        if rule_result:
            return rule_result
        
        # 2) 퍼지 매칭 (고급 기능)
        fuzzy_result = self.fuzzy_category(n, threshold=90)
        if fuzzy_result:
            return fuzzy_result
        
        # 3) AI 머신러닝 분류 (최고급 기능, 사용 가능한 경우)
        if self.ai_model_available and self.ml_classifier and self.label_encoder:
            try:
                features = self._extract_features_batch([n])
                prediction = self.ml_classifier.predict(features)[0]
                return self.label_encoder.inverse_transform([prediction])[0]
            except Exception as e:
                print(f"⚠️ AI 분류 실패, 기본 분류 사용: {e}")
        
        # 4) 기존 샘플 기반 분류 (폴백)
        for category, samples in self.CATEGORIES.items():
            for sample in samples:
                if sample.lower() in n or n in sample.lower():
                    return category
        
        return "기타"
    
    def parse_nh_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """
        NH농협 거래내역 엑셀 파일 파싱 (기존 기능 완전 보존)
        """
        # 모델 초기화 (필요한 경우)
        if not self.model_initialized:
            self.initialize_model()
            
        try:
            # 엑셀 파일 읽기
            raw = pd.read_excel(file_path, header=None)
            
            # '순번' 헤더 찾기
            hdr_rows = raw[raw.apply(lambda r: r.astype(str).str.contains("순번").any(), axis=1)].index
            if len(hdr_rows) == 0:
                raise ValueError("'순번' 헤더를 찾을 수 없습니다. NH농협 거래내역 파일인지 확인해주세요.")
            
            # 헤더를 기준으로 다시 읽기
            df = pd.read_excel(file_path, header=hdr_rows[0])
            
            # 필요한 컬럼 찾기
            want_cols = ["거래일시", "출금금액", "입금금액", "거래후잔액", "거래내용", "거래기록사항"]
            cols = []
            for want in want_cols:
                found_col = None
                for col in df.columns:
                    if want in str(col):
                        found_col = col
                        break
                if found_col:
                    cols.append(found_col)
            
            if not cols:
                raise ValueError("필요한 컬럼을 찾을 수 없습니다.")
            
            df = df[cols].fillna("")
            
            # 출금 거래만 필터링
            출금_컬럼 = None
            for col in df.columns:
                if "출금" in str(col):
                    출금_컬럼 = col
                    break
            
            if 출금_컬럼 is None:
                raise ValueError("출금금액 컬럼을 찾을 수 없습니다.")
            
            df = df[df[출금_컬럼].astype(str).str.strip() != ""]
            df = df[df[출금_컬럼].astype(str).str.strip() != "0"]
            
            # 상호명 추출
            거래기록사항_컬럼 = None
            for col in df.columns:
                if "거래기록사항" in str(col) or "상호명" in str(col):
                    거래기록사항_컬럼 = col
                    break
            
            if 거래기록사항_컬럼 is None:
                # 거래내용 컬럼으로 대체
                for col in df.columns:
                    if "거래내용" in str(col):
                        거래기록사항_컬럼 = col
                        break
            
            if 거래기록사항_컬럼 is None:
                raise ValueError("상호명 정보를 찾을 수 없습니다.")
            
            df["store_name"] = df[거래기록사항_컬럼].astype(str).str.strip()
            
            # 🔥 고급 AI 카테고리 예측 적용
            print(f"🤖 {'고급 AI' if self.ai_model_available else '기본'} 분류 모드로 {len(df)}건 처리 중...")
            df["predicted_category"] = df["store_name"].apply(self.categorize_store)
            
            # 결과 생성
            result = []
            거래일시_컬럼 = None
            for col in df.columns:
                if "거래일시" in str(col) or "거래일" in str(col):
                    거래일시_컬럼 = col
                    break
            
            for _, row in df.iterrows():
                try:
                    # 날짜 파싱
                    transaction_date = datetime.now()
                    if 거래일시_컬럼 and pd.notna(row[거래일시_컬럼]):
                        try:
                            transaction_date = pd.to_datetime(row[거래일시_컬럼])
                        except:
                            pass
                    
                    # 금액 파싱
                    amount_str = str(row[출금_컬럼]).replace(",", "").replace(" ", "")
                    amount = 0
                    try:
                        amount = float(re.sub(r'[^\d.]', '', amount_str))
                    except:
                        continue
                    
                    if amount <= 0:
                        continue
                    
                    result.append({
                        "transaction_date": transaction_date.isoformat() if hasattr(transaction_date, 'isoformat') else str(transaction_date),
                        "amount": amount,
                        "store_name": row["store_name"],
                        "category": row["predicted_category"],
                        "description": str(row[거래기록사항_컬럼]) if 거래기록사항_컬럼 else "",
                        "payment_method": "카드"
                    })
                    
                except Exception as e:
                    print(f"행 처리 중 오류: {e}")
                    continue
            
            print(f"✅ 총 {len(result)}건의 거래를 {'고급 AI' if self.ai_model_available else '기본'}로 분류했습니다.")
            return result
            
        except Exception as e:
            raise Exception(f"NH 거래내역 파싱 실패: {str(e)}")

    def parse_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        일반 CSV 거래내역 파일 파싱 (기존 기능 완전 보존)
        """
        # 모델 초기화 (필요한 경우)
        if not self.model_initialized:
            self.initialize_model()
            
        try:
            # CSV 파일 읽기 (여러 인코딩 시도)
            encodings = ['utf-8', 'cp949', 'euc-kr']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ {encoding} 인코딩으로 파일 읽기 성공")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("지원하는 인코딩으로 파일을 읽을 수 없습니다.")
            
            print(f"📊 CSV 파일 컬럼: {list(df.columns)}")
            print(f"📊 총 {len(df)}행의 데이터")
            
            # 컬럼명 매핑 (다양한 형태의 컬럼명 지원)
            date_columns = ['날짜', '거래일', '거래일시', 'date', '일자']
            amount_columns = ['금액', '출금액', '입금액', '출금금액', '입금금액', '사용금액', 'amount', '지출액']
            merchant_columns = ['가맹점', '가맹점명', '상호명', '적요', 'merchant', '거래처', '내용']
            
            # 실제 컬럼 찾기
            date_col = None
            amount_col = None
            merchant_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if not date_col:
                    for date_keyword in date_columns:
                        if date_keyword.lower() in col_lower:
                            date_col = col
                            break
                
                if not amount_col:
                    for amount_keyword in amount_columns:
                        if amount_keyword.lower() in col_lower:
                            amount_col = col
                            break
                
                if not merchant_col:
                    for merchant_keyword in merchant_columns:
                        if merchant_keyword.lower() in col_lower:
                            merchant_col = col
                            break
            
            print(f"🔍 감지된 컬럼 - 날짜: {date_col}, 금액: {amount_col}, 가맹점: {merchant_col}")
            
            # 필수 컬럼 확인
            if not amount_col:
                raise ValueError("금액 컬럼을 찾을 수 없습니다. (금액, 출금액, 사용금액 등)")
            
            if not merchant_col:
                raise ValueError("가맹점 컬럼을 찾을 수 없습니다. (가맹점명, 상호명, 적요 등)")
            
            # 데이터 처리
            result = []
            
            # 🔥 배치 방식으로 고급 AI 분류 (성능 최적화)
            valid_merchants = []
            valid_indices = []
            
            for index, row in df.iterrows():
                try:
                    # 금액 검증
                    amount_str = str(row[amount_col]).replace(",", "").replace(" ", "")
                    if pd.isna(row[amount_col]) or amount_str == '' or amount_str == 'nan':
                        continue
                    
                    amount = float(re.sub(r'[^\d.]', '', amount_str))
                    if amount <= 0:
                        continue
                    
                    # 가맹점명 검증
                    merchant_name = str(row[merchant_col]) if pd.notna(row[merchant_col]) else ""
                    if merchant_name.strip() == '' or merchant_name == 'nan':
                        merchant_name = "기타"
                    
                    valid_merchants.append(merchant_name)
                    valid_indices.append(index)
                    
                except:
                    continue
            
            # 배치 카테고리 분류
            print(f"🤖 {len(valid_merchants)}개 상호명에 대해 {'고급 AI' if self.ai_model_available else '기본'} 분류 진행...")
            categories = [self.categorize_store(merchant) for merchant in valid_merchants]
            
            # 결과 생성
            for i, index in enumerate(valid_indices):
                try:
                    row = df.iloc[index]
                    
                    # 날짜 파싱
                    transaction_date = datetime.now()
                    if date_col and pd.notna(row[date_col]):
                        try:
                            transaction_date = pd.to_datetime(row[date_col])
                        except:
                            pass
                    
                    # 금액 파싱
                    amount_str = str(row[amount_col]).replace(",", "").replace(" ", "")
                    amount = float(re.sub(r'[^\d.]', '', amount_str))
                    
                    result.append({
                        "transaction_date": transaction_date.isoformat() if hasattr(transaction_date, 'isoformat') else str(transaction_date),
                        "amount": amount,
                        "store_name": valid_merchants[i],
                        "category": categories[i],
                        "description": valid_merchants[i],
                        "payment_method": "카드"
                    })
                    
                except Exception as e:
                    print(f"행 {index} 처리 중 오류: {e}")
                    continue
            
            print(f"✅ 총 {len(result)}건의 거래를 {'고급 AI' if self.ai_model_available else '기본'}로 분류했습니다.")
            return result
            
        except Exception as e:
            raise Exception(f"CSV 파일 파싱 실패: {str(e)}")


# 전역 인스턴스 생성
classification_service = TransactionClassificationService()