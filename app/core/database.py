# app/core/database.py
from mysql.connector import MySQLConnection, connect
from mysql.connector import Error

def get_db_connection():
    """
    MySQL 데이터베이스 연결을 생성하고 반환하는 함수
    
    Returns:
        MySQLConnection: 성공적으로 연결된 경우 연결 객체, 실패 시 None
    """
    # 테스트 모드 코드를 주석 처리하거나 제거
    # print("⚠️ 데이터베이스 연결 시도를 건너뜁니다 (테스트 모드)")
    # return None
    
    try:
        connection = connect(
            host="localhost",
            user="root",  # 필요한 경우 변경
            password="1234",  # 필요한 경우 변경
            database="finance_app"
        )
        if connection.is_connected():
            print("✅ 성공적으로 MySQL에 연결했습니다")
        return connection
    except Error as e:
        print("🚨 MySQL 연결 오류:", e)
        return None