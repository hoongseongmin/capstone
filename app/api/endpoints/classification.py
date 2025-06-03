# app/api/endpoints/classification.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any
from mysql.connector import MySQLConnection
import tempfile
import os
import pandas as pd
import json

from app.core.database import get_db_connection
from app.services.transaction_service import TransactionService

# API 라우터 생성
router = APIRouter()

@router.post("/upload-nh-excel", response_model=Dict[str, Any])
async def upload_nh_excel(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    NH농협 거래내역 엑셀 파일을 업로드하고 자동 분류하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        file: 업로드할 NH 거래내역 엑셀 파일
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 처리 결과
    """
    from app.services.classification_service import classification_service
    
    if db is None:
        # 데이터베이스 연결이 없어도 로컬 처리는 가능
        print("⚠️ 데이터베이스 연결 없음 - 로컬 처리만 수행")
    
    # 파일 형식 검증
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="엑셀 파일(.xlsx, .xls)만 업로드 가능합니다.")
    
    temp_file_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # AI 모델 초기화 (처음 사용시에만)
        print("🤖 분류 서비스 초기화 중...")
        classification_service.initialize_model()
        
        # NH 거래내역 파싱 및 분류
        print("📊 NH 거래내역 파싱 및 분류 중...")
        transactions_data = classification_service.parse_nh_excel(temp_file_path)
        
        # 데이터베이스 저장 시도 (연결이 있을 때만)
        processing_result = None
        if db is not None:
            try:
                service = TransactionService(db)
                processing_result = service.process_ai_analyzed_transactions(user_id, transactions_data)
                db.close()
            except Exception as db_error:
                print(f"⚠️ 데이터베이스 저장 실패: {db_error}")
                # 데이터베이스 저장 실패해도 계속 진행
        
        return {
            "success": True,
            "message": "NH 거래내역이 성공적으로 업로드되고 분류되었습니다.",
            "file_name": file.filename,
            "total_transactions": len(transactions_data),
            "processing_result": processing_result,
            "transactions_preview": transactions_data[:5] if transactions_data else [],
            "categories_found": list(set([t.get('category', '기타') for t in transactions_data]))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")
        
    finally:
        # 임시 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        if db is not None:
            try:
                db.close()
            except:
                pass

@router.post("/classify-merchant", response_model=Dict[str, Any])
async def classify_single_merchant(
    merchant_name: str = Form(...)
):
    """
    단일 가맹점명 분류
    
    Args:
        merchant_name: 분류할 가맹점명
        
    Returns:
        Dict[str, Any]: 분류 결과
    """
    try:
        from app.services.classification_service import classification_service
        
        classification_service.initialize_model()
        category = classification_service.categorize_store(merchant_name)
        
        return {
            "merchant_name": merchant_name,
            "category": category,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분류 중 오류 발생: {str(e)}")

@router.get("/categories", response_model=Dict[str, Any])
def get_available_categories():
    """
    사용 가능한 카테고리 목록 반환
    
    Returns:
        Dict[str, Any]: 카테고리 목록
    """
    from app.services.classification_service import classification_service
    
    return {
        "categories": list(classification_service.CATEGORIES.keys()),
        "total_count": len(classification_service.CATEGORIES)
    }

@router.post("/process-excel-local", response_model=Dict[str, Any])
async def process_excel_local(
    file: UploadFile = File(...)
):
    """
    데이터베이스 없이 로컬에서만 엑셀 파일을 처리하고 분류하는 엔드포인트
    
    Args:
        file: 업로드할 거래내역 엑셀 파일 (.xlsx, .xls, .csv)
    
    Returns:
        Dict[str, Any]: 처리 결과 (분류된 거래 데이터)
    """
    from app.services.classification_service import classification_service
    
    # 파일 형식 검증
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(status_code=400, detail="엑셀 파일(.xlsx, .xls) 또는 CSV 파일(.csv)만 업로드 가능합니다.")
    
    temp_file_path = None
    try:
        # 파일 확장자에 따른 임시 파일 생성
        file_suffix = '.csv' if file.filename.endswith('.csv') else '.xlsx'
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 분류 서비스 초기화
        print("🤖 분류 서비스 초기화 중...")
        classification_service.initialize_model()
        
        # 파일 타입에 따른 처리
        if file.filename.endswith('.csv'):
            print("📊 CSV 파일 파싱 및 분류 중...")
            transactions_data = classification_service.parse_csv_file(temp_file_path)
        else:
            print("📊 NH 거래내역 파싱 및 분류 중...")
            transactions_data = classification_service.parse_nh_excel(temp_file_path)
        
        # 카테고리별 통계 생성
        category_stats = {}
        total_amount = 0
        
        for transaction in transactions_data:
            category = transaction.get('category', '기타')
            amount = transaction.get('amount', 0)
            
            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'total_amount': 0,
                    'transactions': []
                }
            
            category_stats[category]['count'] += 1
            category_stats[category]['total_amount'] += amount
            category_stats[category]['transactions'].append(transaction)
            total_amount += amount
        
        # 카테고리별 비율 계산
        for category in category_stats:
            if total_amount > 0:
                category_stats[category]['percentage'] = round(
                    (category_stats[category]['total_amount'] / total_amount) * 100, 2
                )
            else:
                category_stats[category]['percentage'] = 0
        
        return {
            "success": True,
            "message": f"파일이 성공적으로 처리되고 분류되었습니다.",
            "file_name": file.filename,
            "file_type": "CSV" if file.filename.endswith('.csv') else "Excel",
            "total_transactions": len(transactions_data),
            "total_amount": total_amount,
            "categories_summary": {
                category: {
                    "count": stats["count"],
                    "total_amount": stats["total_amount"],
                    "percentage": stats["percentage"]
                }
                for category, stats in category_stats.items()
            },
            "transactions_preview": transactions_data[:10] if transactions_data else [],
            "categories_found": list(category_stats.keys()),
            "all_transactions": transactions_data  # 모든 거래 데이터 반환
        }
        
    except Exception as e:
        error_message = f"파일 처리 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        raise HTTPException(status_code=500, detail=error_message)
        
    finally:
        # 임시 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print("🗑️ 임시 파일 정리 완료")
            except Exception as cleanup_error:
                print(f"⚠️ 임시 파일 정리 실패: {cleanup_error}")

@router.get("/test-local")
def test_local_service():
    """
    로컬 분류 서비스 테스트 엔드포인트
    
    Returns:
        Dict[str, Any]: 테스트 결과
    """
    try:
        from app.services.classification_service import classification_service
        
        classification_service.initialize_model()
        
        # 테스트 가맹점명들
        test_merchants = [
            "스타벅스 강남점",
            "맥도날드 홍대점",
            "GS25 편의점",
            "카카오택시",
            "한국전력공사",
            "SKT",
            "서울대병원",
            "YBM어학원",
            "넷플릭스",
            "CGV 영화관"
        ]
        
        test_results = []
        for merchant in test_merchants:
            category = classification_service.categorize_store(merchant)
            test_results.append({
                "merchant_name": merchant,
                "predicted_category": category
            })
        
        return {
            "success": True,
            "message": "로컬 분류 서비스가 정상 작동합니다.",
            "available_categories": list(classification_service.CATEGORIES.keys()),
            "test_results": test_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"테스트 중 오류 발생: {str(e)}")

@router.post("/batch-classify", response_model=Dict[str, Any])
async def batch_classify_merchants(
    merchants: List[str] = Form(..., description="분류할 가맹점명 목록")
):
    """
    여러 가맹점명을 한번에 분류하는 엔드포인트
    
    Args:
        merchants: 분류할 가맹점명 리스트
        
    Returns:
        Dict[str, Any]: 배치 분류 결과
    """
    try:
        from app.services.classification_service import classification_service
        
        classification_service.initialize_model()
        
        results = []
        category_counts = {}
        
        for merchant in merchants:
            category = classification_service.categorize_store(merchant)
            results.append({
                "merchant_name": merchant,
                "predicted_category": category
            })
            
            # 카테고리별 카운트
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        return {
            "success": True,
            "message": f"{len(merchants)}개 가맹점명 분류 완료",
            "total_processed": len(merchants),
            "results": results,
            "category_distribution": category_counts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 분류 중 오류 발생: {str(e)}")

@router.post("/save-to-database", response_model=Dict[str, Any])
async def save_transactions_to_database(
    user_id: int = Form(...),
    transactions_data: str = Form(..., description="JSON 형태의 거래 데이터"),
    db: MySQLConnection = Depends(get_db_connection)
):
    """
    분류된 거래 데이터를 데이터베이스에 저장하는 엔드포인트
    
    Args:
        user_id: 사용자 ID
        transactions_data: JSON 형태의 거래 데이터
        db: 데이터베이스 연결 객체
    
    Returns:
        Dict[str, Any]: 저장 결과
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # JSON 파싱
        transactions = json.loads(transactions_data)
        
        # 거래 서비스를 통해 데이터베이스에 저장
        service = TransactionService(db)
        processing_result = service.process_ai_analyzed_transactions(user_id, transactions)
        
        db.close()
        return processing_result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="잘못된 JSON 형식입니다.")
    except Exception as e:
        if db:
            db.close()
        raise HTTPException(status_code=500, detail=f"데이터베이스 저장 중 오류 발생: {str(e)}")

@router.get("/classification-stats", response_model=Dict[str, Any])
def get_classification_statistics():
    """
    분류 서비스의 통계 정보를 반환하는 엔드포인트
    
    Returns:
        Dict[str, Any]: 분류 통계
    """
    try:
        from app.services.classification_service import classification_service
        
        stats = {
            "total_categories": len(classification_service.CATEGORIES),
            "categories_detail": {
                category: {
                    "keywords_count": len(keywords),
                    "sample_keywords": keywords[:5]  # 처음 5개만 표시
                }
                for category, keywords in classification_service.CATEGORIES.items()
            },
            "rule_based_categories": len(classification_service.RULES),
            "rules_detail": [
                {
                    "category": rule[0],
                    "keywords": rule[1]
                }
                for rule in classification_service.RULES
            ],
            "model_initialized": classification_service.model_initialized
        }
        
        return {
            "success": True,
            "message": "분류 서비스 통계 정보",
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 정보 조회 중 오류 발생: {str(e)}")

@router.post("/process-complete", response_model=Dict[str, Any])
async def process_complete_analysis(
    file: UploadFile = File(...),
    manual_column_mapping: str = Form(None, description="JSON 형태의 수동 컬럼 매핑")
):
    """
    🚀 통합 거래 분석 서비스 - 파일 업로드부터 고도화된 분류까지 원스톱 처리
    
    Args:
        file: 업로드할 파일 (.xlsx, .xls, .csv)
        manual_column_mapping: JSON 형태의 수동 컬럼 매핑 (선택적)
    
    Returns:
        Dict[str, Any]: 완전한 분석 결과
    """
    from app.services.enhanced_classification_service import enhanced_classification_service
    
    if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
        raise HTTPException(status_code=400, detail="엑셀 파일(.xlsx, .xls) 또는 CSV 파일(.csv)만 업로드 가능합니다.")
    
    temp_file_path = None
    used_encoding = "UTF-8"  # 기본값 설정
    
    try:
        # 1단계: 파일 저장
        file_suffix = '.csv' if file.filename.endswith('.csv') else '.xlsx'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print("🔍 1단계: 파일 구조 분석 중...")
        
        # 2단계: 파일 읽기 및 구조 분석
        try:
            if file.filename.endswith('.csv'):
                # CSV 파일의 경우 여러 인코딩 시도
                encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(temp_file_path, encoding=encoding)
                        used_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("지원하는 인코딩으로 파일을 읽을 수 없습니다.")
                    
                print(f"✅ CSV 파일 읽기 성공 ({used_encoding} 인코딩)")
            else:
                # Excel 파일 처리
                df = pd.read_excel(temp_file_path)
                used_encoding = "UTF-8"
                print("✅ Excel 파일 읽기 성공")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"파일 읽기 실패: {str(e)}")
        
        # 3단계: 서비스 초기화
        enhanced_classification_service.initialize_model()
        
        # 4단계: 파일 구조 자동 분석
        file_structure = enhanced_classification_service.analyze_file_structure(df)
        print(f"📊 파일 구조 분석 완료 - {file_structure['total_rows']}행, {file_structure['total_columns']}컬럼")
        
        # 5단계: 수동 컬럼 매핑 적용 (있는 경우)
        if manual_column_mapping:
            try:
                manual_mapping = json.loads(manual_column_mapping)
                file_structure["detected_columns"].update(manual_mapping)
                print("✅ 수동 컬럼 매핑 적용됨")
            except json.JSONDecodeError:
                print("⚠️ 수동 컬럼 매핑 형식 오류 - 무시하고 계속 진행")
        
        # 6단계: 고도화된 거래 처리 및 분석
        print("🤖 고도화된 거래 분류 및 분석 중...")
        processing_result = enhanced_classification_service.process_transactions_with_analysis(df, file_structure)
        
        # 7단계: 기타 거래 심층 분석
        unclassified_transactions = [
            t for t in processing_result["transactions"] 
            if t["category"] == "기타"
        ]
        
        unclassified_analysis = None
        if unclassified_transactions:
            print(f"🔍 기타 거래 {len(unclassified_transactions)}건 분석 중...")
            # 간단한 분석 (business_search_service 없이)
            unclassified_analysis = {
                "total_unclassified": len(unclassified_transactions),
                "unclassified_merchants": list(set([t["store_name"] for t in unclassified_transactions]))[:20],
                "total_amount": sum([t["amount"] for t in unclassified_transactions]),
                "avg_amount": sum([t["amount"] for t in unclassified_transactions]) / len(unclassified_transactions),
                "recommendations": [
                    "💡 기타로 분류된 거래들을 수동으로 검토하여 새로운 키워드를 추가하세요.",
                    "🔍 반복적으로 나타나는 가맹점명을 확인하여 카테고리 룰을 개선하세요."
                ]
            }
        
        # 8단계: 종합 결과 생성
        result = {
            "success": True,
            "message": f"파일 처리 완료 - {processing_result['analysis']['total_transactions']}건 분석",
            "file_info": {
                "name": file.filename,
                "type": "CSV" if file.filename.endswith('.csv') else "Excel",
                "encoding": used_encoding,
            },
            "file_structure_analysis": file_structure,
            "processing_result": processing_result,
            "unclassified_deep_analysis": unclassified_analysis,
            "overall_summary": {
                "total_transactions": processing_result["analysis"]["total_transactions"],
                "total_amount": processing_result["analysis"]["total_amount"],
                "classification_success_rate": round(100 - processing_result["analysis"]["unclassified_analysis"]["percentage"], 2),
                "avg_confidence": round(
                    sum(t["classification_confidence"] for t in processing_result["transactions"]) / 
                    len(processing_result["transactions"]), 2
                ) if processing_result["transactions"] else 0,
                "top_categories": sorted(
                    processing_result["analysis"]["category_breakdown"].items(),
                    key=lambda x: x[1]["amount"],
                    reverse=True
                )[:5]
            },
            "action_items": processing_result["recommendations"] + (
                [f"📋 기타 거래 {len(unclassified_transactions)}건에 대한 추가 분석 결과를 확인하세요."] 
                if unclassified_transactions else []
            )
        }
        
        print("✅ 모든 처리 완료!")
        return result
        
    except Exception as e:
        error_message = f"파일 처리 중 오류 발생: {str(e)}"
        print(f"❌ {error_message}")
        raise HTTPException(status_code=500, detail=error_message)
        
    finally:
        # 임시 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print("🗑️ 임시 파일 정리 완료")
            except:
                pass

@router.post("/extract-merchant-test")
async def test_merchant_extraction(
    descriptions: List[str] = Form(..., description="거래기록사항 목록")
):
    """
    거래기록사항에서 가맹점명 추출 테스트
    
    Args:
        descriptions: 테스트할 거래기록사항 목록
        
    Returns:
        Dict[str, Any]: 추출 결과
    """
    try:
        from app.services.enhanced_classification_service import enhanced_classification_service
        
        enhanced_classification_service.initialize_model()
        
        results = []
        for desc in descriptions:
            extracted = enhanced_classification_service.extract_merchant_from_description(desc)
            classification = enhanced_classification_service.advanced_categorize(extracted, desc)
            
            results.append({
                "original_description": desc,
                "extracted_merchant": extracted,
                "predicted_category": classification["category"],
                "confidence": classification["confidence"],
                "method": classification["method"]
            })
        
        return {
            "success": True,
            "results": results,
            "summary": {
                "total_tested": len(results),
                "extraction_success_rate": round(
                    sum(1 for r in results if r["extracted_merchant"] != "미상") / len(results) * 100, 2
                ) if results else 0,
                "classification_distribution": {
                    category: sum(1 for r in results if r["predicted_category"] == category)
                    for category in set(r["predicted_category"] for r in results)
                },
                "high_confidence_predictions": sum(1 for r in results if r["confidence"] >= 80),
                "avg_confidence": round(
                    sum(r["confidence"] for r in results) / len(results), 2
                ) if results else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"테스트 중 오류 발생: {str(e)}")

@router.get("/health-check")
def health_check():
    """
    분류 서비스 상태 확인 엔드포인트
    
    Returns:
        Dict[str, Any]: 서비스 상태
    """
    try:
        from app.services.classification_service import classification_service
        from app.services.enhanced_classification_service import enhanced_classification_service
        
        basic_service_status = "OK" if classification_service else "ERROR"
        enhanced_service_status = "OK" if enhanced_classification_service else "ERROR"
        
        return {
            "success": True,
            "message": "분류 서비스 상태 확인",
            "services": {
                "basic_classification": {
                    "status": basic_service_status,
                    "initialized": getattr(classification_service, 'model_initialized', False)
                },
                "enhanced_classification": {
                    "status": enhanced_service_status,
                    "initialized": getattr(enhanced_classification_service, 'model_initialized', False)
                }
            },
            "available_endpoints": [
                "/classification/upload-nh-excel",
                "/classification/process-excel-local", 
                "/classification/process-complete",
                "/classification/classify-merchant",
                "/classification/batch-classify",
                "/classification/categories",
                "/classification/test-local",
                "/classification/extract-merchant-test",
                "/classification/health-check"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"서비스 상태 확인 중 오류: {str(e)}",
            "services": {
                "basic_classification": {"status": "ERROR"},
                "enhanced_classification": {"status": "ERROR"}
            }
        }