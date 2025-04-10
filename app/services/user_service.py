# app/services/user_service.py
from typing import List, Optional, Dict, Any
import hashlib
import os
from mysql.connector import MySQLConnection

from app.models.user import User
from app.repositories.user_repository import UserRepository

class UserService:
    """
    사용자 관련 비즈니스 로직을 처리하는 서비스 클래스
    
    이 클래스는 Repository 계층을 사용하여 사용자 정보를 관리합니다.
    """
    
    def __init__(self, connection: MySQLConnection):
        """
        UserService 초기화
        
        Args:
            connection: MySQL 데이터베이스 연결 객체
        """
        self.repository = UserRepository(connection)
    
    def create_user(self, username: str, password: str, name: str, age: int, occupation: str, 
                   address: str, gender: str, contact: str, income_level: Optional[str] = None) -> int:
        """
        새로운 사용자를 생성
        
        Args:
            username: 사용자 로그인 ID
            password: 비밀번호
            name: 이름
            age: 나이
            occupation: 직업
            address: 거주 지역
            gender: 성별
            contact: 연락처
            income_level: 소득 수준 (선택적)
        
        Returns:
            int: 생성된 사용자의 ID
        """
        # 비밀번호 해싱
        password_hash = self._hash_password(password)
        
        # 사용자 객체 생성
        user = User(
            id=None,
            username=username,
            password_hash=password_hash,
            name=name,
            age=age,
            occupation=occupation,
            address=address,
            gender=gender,
            contact=contact,
            income_level=income_level
        )
        
        return self.repository.create(user)
    
    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        사용자 인증
        
        Args:
            username: 사용자 로그인 ID
            password: 비밀번호
        
        Returns:
            Optional[Dict[str, Any]]: 인증 성공 시 사용자 정보, 실패 시 None
        """
        user = self.repository.get_by_username(username)
        
        if not user:
            return None
        
        password_hash = self._hash_password(password)
        
        if user['password_hash'] != password_hash:
            return None
        
        # 비밀번호 해시 정보는 응답에서 제외
        user.pop('password_hash', None)
        return user
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        모든 사용자 목록을 가져옴
        
        Returns:
            List[Dict[str, Any]]: 사용자 정보 딕셔너리 리스트
        """
        users = self.repository.get_all()
        
        # 비밀번호 해시 정보는 응답에서 제외
        for user in users:
            user.pop('password_hash', None)
            
        return users
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        지정된 ID의 사용자를 가져옴
        
        Args:
            user_id: 찾을 사용자 ID
        
        Returns:
            Optional[Dict[str, Any]]: 찾은 경우 사용자 정보 딕셔너리, 찾지 못한 경우 None
        """
        user = self.repository.get_by_id(user_id)
        
        if user:
            # 비밀번호 해시 정보는 응답에서 제외
            user.pop('password_hash', None)
            
        return user
    
    def update_user(self, user_id: int, user_data: Dict[str, Any]) -> bool:
        """
        지정된 ID의 사용자 정보를 업데이트
        
        Args:
            user_id: 업데이트할 사용자 ID
            user_data: 업데이트할 데이터 딕셔너리
        
        Returns:
            bool: 업데이트 성공 여부
        """
        # 비밀번호가 포함된 경우 해싱
        if 'password' in user_data:
            user_data['password_hash'] = self._hash_password(user_data.pop('password'))
            
        return self.repository.update(user_id, user_data)
    
    def delete_user(self, user_id: int) -> bool:
        """
        지정된 ID의 사용자를 삭제
        
        Args:
            user_id: 삭제할 사용자 ID
        
        Returns:
            bool: 삭제 성공 여부
        """
        return self.repository.delete(user_id)
    
    def create_users_table(self) -> None:
        """
        사용자 테이블을 생성 (존재하지 않는 경우)
        """
        self.repository.create_table()
    
    def _hash_password(self, password: str) -> str:
        """
        비밀번호를 해싱
        
        Args:
            password: 원본 비밀번호
        
        Returns:
            str: 해싱된 비밀번호
        """
        # 간단한 SHA-256 해싱 (실제 환경에서는 더 강력한 방식 사용 권장)
        return hashlib.sha256(password.encode()).hexdigest()