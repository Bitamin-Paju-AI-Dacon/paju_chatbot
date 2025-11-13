"""
JWT 인증 클래스
"""
from rest_framework import authentication, exceptions
from django.contrib.auth import get_user_model
from app.auth.utils import verify_token

User = get_user_model()


class JWTAuthentication(authentication.BaseAuthentication):
    """JWT Bearer Token 인증"""
    
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        
        if not auth_header:
            return None
        
        try:
            # "Bearer {token}" 형식에서 토큰 추출
            token_type, token = auth_header.split(' ', 1)
            if token_type.lower() != 'bearer':
                return None
        except ValueError:
            return None
        
        try:
            # 토큰 검증
            payload = verify_token(token)
            username = payload.get('sub')
            user_id = payload.get('user_id')
            
            if not username or not user_id:
                raise exceptions.AuthenticationFailed('토큰이 유효하지 않습니다')
            
            # 사용자 조회
            try:
                user = User.objects.get(id=user_id, username=username, is_active=True)
            except User.DoesNotExist:
                raise exceptions.AuthenticationFailed('사용자를 찾을 수 없습니다')
            
            return (user, token)
            
        except exceptions.APIException:
            raise
        except Exception:
            raise exceptions.AuthenticationFailed('토큰이 유효하지 않습니다')

