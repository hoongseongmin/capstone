# 동일 집단 소비습관 분석 API 📊

<div align="center">
  
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

## 📋 개요

이 애플리케이션은 사용자의 소비 패턴을 분석하고 인사이트를 제공하는 종합 재무 관리 시스템입니다. 유사한 인구통계학적 특성을 가진 집단 내에서 소비 패턴을 비교하고 분석하여 사용자에게 의미 있는 금융 인사이트를 제공합니다.

## 🚀 주요 기능

- ✅ **사용자 관리** - 회원가입, 로그인, 프로필 관리
- 💰 **거래 기록 관리** - 수입 및 지출 내역 추적
- 📈 **고급 소비 패턴 분석** - 시간별, 카테고리별 소비 분석
- 👥 **동일 집단 소비 비교** - 비슷한 인구 특성을 가진 그룹과 비교
- 📅 **월별 소비 추세 분석** - 시간에 따른 소비 패턴 변화 추적
- ⚠️ **이상 소비 패턴 감지** - 비정상적인 지출 패턴 알림

## 🛠️ 기술 스택

- **백엔드**: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
- **데이터베이스**: ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
- **인증**: ![JWT](https://img.shields.io/badge/JWT-000000?style=flat-square&logo=json-web-tokens&logoColor=white)
- **분석**: Python 통계 라이브러리

## 🔌 API 엔드포인트

### 사용자 관리
| 메소드 | 엔드포인트 | 설명 |
|--------|------------|------|
| POST | `/users` | 사용자 생성 |
| POST | `/users/login` | 로그인 |
| GET | `/users/{user_id}` | 사용자 정보 조회 |
| PUT | `/users/{user_id}` | 사용자 정보 업데이트 |

### 거래 관리
| 메소드 | 엔드포인트 | 설명 |
|--------|------------|------|
| POST | `/transactions` | 거래 생성 |
| GET | `/transactions/user/{user_id}` | 사용자 거래 내역 조회 |
| DELETE | `/transactions/{transaction_id}` | 거래 삭제 |

### 분석
| 메소드 | 엔드포인트 | 설명 |
|--------|------------|------|
| GET | `/analysis/spending/{user_id}` | 개인 소비 패턴 분석 |
| GET | `/analysis/group-comparison/{user_id}` | 동일 집단 비교 |
| GET | `/analysis/insights/{user_id}` | 개인화된 소비 인사이트 |

## 📊 데이터 모델

### User
| 필드 | 타입 | 설명 |
|------|------|------|
| id | int | 사용자 고유 식별자 |
| username | string | 사용자 로그인 ID |
| name | string | 사용자 이름 |
| age | int | 사용자 나이 |
| occupation | string | 사용자 직업 |
| address | string | 사용자 주소 |

### Transaction
| 필드 | 타입 | 설명 |
|------|------|------|
| id | int | 거래 고유 식별자 |
| user_id | int | 사용자 ID (외래 키) |
| category_id | int | 카테고리 ID (외래 키) |
| amount | float | 거래 금액 |
| transaction_date | datetime | 거래 날짜 |

### TransactionCategory
| 필드 | 타입 | 설명 |
|------|------|------|
| id | int | 카테고리 고유 식별자 |
| name | string | 카테고리 이름 |
| description | string | 카테고리 설명 |


## 🔧 개발 환경 설정


### 필수 요구사항
- Python 3.8+
- MySQL



### 설치 단계

```bash
# 저장소 클론
git clone https://github.com/your-username/consume-analysis-app.git
cd consume-analysis-app

# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# Windows:
.\venv\Scripts\activate

# 추가 패키지 설치
pip install pyjwt
pip install pydantic-settings
