import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from database import Database
from config import JWT_CONFIG

logger = logging.getLogger(__name__)

class Auth:
    def __init__(self, database: Database):
        self.db = database
        self.jwt_secret = JWT_CONFIG['secret_key']
        self.jwt_algorithm = JWT_CONFIG['algorithm']
        self.jwt_expire_minutes = JWT_CONFIG['expire_minutes']

    def register_user(self, username: str, password: str, email: str, 
                     role: str = 'user') -> bool:
        """注册新用户"""
        try:
            # 检查用户名是否已存在
            if self.db.get_user_by_username(username):
                logger.warning(f"Username {username} already exists")
                return False

            # 密码加密
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

            # 创建用户记录
            user_data = {
                'username': username,
                'password': hashed_password.decode('utf-8'),
                'email': email,
                'role': role,
                'created_at': datetime.now(),
                'last_login': None,
                'is_active': True
            }

            return self.db.create_user(user_data)
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False

    def login(self, username: str, password: str) -> Optional[Dict]:
        """用户登录"""
        try:
            # 获取用户信息
            user = self.db.get_user_by_username(username)
            if not user:
                logger.warning(f"User {username} not found")
                return None

            # 验证密码
            if not bcrypt.checkpw(password.encode('utf-8'), 
                                user['password'].encode('utf-8')):
                logger.warning(f"Invalid password for user {username}")
                return None

            # 检查用户状态
            if not user['is_active']:
                logger.warning(f"User {username} is inactive")
                return None

            # 更新最后登录时间
            self.db.update_user_last_login(username)

            # 生成JWT令牌
            token = self._generate_token(user)

            return {
                'token': token,
                'user': {
                    'username': user['username'],
                    'email': user['email'],
                    'role': user['role']
                }
            }
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return None

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    def _generate_token(self, user: Dict) -> str:
        """生成JWT令牌"""
        try:
            payload = {
                'sub': user['username'],
                'role': user['role'],
                'exp': datetime.utcnow() + timedelta(minutes=self.jwt_expire_minutes)
            }
            return jwt.encode(
                payload,
                self.jwt_secret,
                algorithm=self.jwt_algorithm
            )
        except Exception as e:
            logger.error(f"Error generating token: {str(e)}")
            raise

    def change_password(self, username: str, old_password: str, 
                       new_password: str) -> bool:
        """修改密码"""
        try:
            # 验证用户和旧密码
            user = self.db.get_user_by_username(username)
            if not user or not bcrypt.checkpw(
                old_password.encode('utf-8'),
                user['password'].encode('utf-8')
            ):
                return False

            # 加密新密码
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(
                new_password.encode('utf-8'),
                salt
            )

            # 更新密码
            return self.db.update_user_password(
                username,
                hashed_password.decode('utf-8')
            )
        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            return False

    def reset_password(self, email: str) -> bool:
        """重置密码"""
        try:
            # 获取用户信息
            user = self.db.get_user_by_email(email)
            if not user:
                return False

            # 生成重置令牌
            reset_token = self._generate_reset_token(user['username'])

            # 发送重置邮件
            self._send_reset_email(user['email'], reset_token)

            return True
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            return False

    def _generate_reset_token(self, username: str) -> str:
        """生成密码重置令牌"""
        try:
            payload = {
                'sub': username,
                'type': 'reset',
                'exp': datetime.utcnow() + timedelta(hours=1)
            }
            return jwt.encode(
                payload,
                self.jwt_secret,
                algorithm=self.jwt_algorithm
            )
        except Exception as e:
            logger.error(f"Error generating reset token: {str(e)}")
            raise

    def _send_reset_email(self, email: str, token: str):
        """发送密码重置邮件"""
        try:
            # 这里需要实现邮件发送逻辑
            # 可以使用之前创建的邮件配置
            pass
        except Exception as e:
            logger.error(f"Error sending reset email: {str(e)}")

    def check_permission(self, user: Dict, required_role: str) -> bool:
        """检查用户权限"""
        try:
            # 角色等级定义
            role_levels = {
                'admin': 3,
                'manager': 2,
                'user': 1
            }

            # 检查用户角色等级是否满足要求
            user_level = role_levels.get(user['role'], 0)
            required_level = role_levels.get(required_role, 0)

            return user_level >= required_level
        except Exception as e:
            logger.error(f"Error checking permission: {str(e)}")
            return False

    def get_user_permissions(self, username: str) -> List[str]:
        """获取用户权限列表"""
        try:
            user = self.db.get_user_by_username(username)
            if not user:
                return []

            # 根据用户角色返回权限列表
            permissions = {
                'admin': ['read', 'write', 'execute', 'manage_users', 'manage_system'],
                'manager': ['read', 'write', 'execute', 'manage_users'],
                'user': ['read', 'execute']
            }

            return permissions.get(user['role'], [])
        except Exception as e:
            logger.error(f"Error getting user permissions: {str(e)}")
            return [] 