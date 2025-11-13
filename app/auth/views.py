"""
인증 API 뷰
"""
from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from datetime import timedelta
from app.auth.serializers import (
    UserCreateSerializer,
    UserSerializer,
    LoginSerializer
)
from app.auth.utils import create_access_token

User = get_user_model()


class SignupView(generics.CreateAPIView):
    """회원가입 API"""
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # 사용자명 중복 확인
        if User.objects.filter(username=serializer.validated_data['username']).exists():
            return Response(
                {"detail": "이미 사용 중인 사용자명입니다"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 이메일 중복 확인
        if User.objects.filter(email=serializer.validated_data['email']).exists():
            return Response(
                {"detail": "이미 사용 중인 이메일입니다"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user = serializer.save()
        return Response(
            UserSerializer(user).data,
            status=status.HTTP_201_CREATED
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """로그인 API"""
    serializer = LoginSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    
    username = serializer.validated_data['username']
    password = serializer.validated_data['password']
    
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response(
            {"detail": "사용자명 또는 비밀번호가 올바르지 않습니다"},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # 비밀번호 검증
    if not user.check_password(password):
        return Response(
            {"detail": "사용자명 또는 비밀번호가 올바르지 않습니다"},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # 활성 사용자 확인
    if not user.is_active:
        return Response(
            {"detail": "비활성화된 계정입니다"},
            status=status.HTTP_403_FORBIDDEN
        )
    
    # JWT 토큰 생성
    access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30일
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    return Response({
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserSerializer(user).data
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me_view(request):
    """현재 사용자 정보 조회"""
    return Response(UserSerializer(request.user).data)

