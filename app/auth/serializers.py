"""
인증 시리얼라이저
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class UserCreateSerializer(serializers.ModelSerializer):
    """회원가입 시리얼라이저"""
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'is_active', 'created_at')
        read_only_fields = ('id', 'is_active', 'created_at')

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(
            password=password,
            **validated_data
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    """사용자 정보 시리얼라이저"""
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'is_active', 'created_at')
        read_only_fields = ('id', 'created_at')


class LoginSerializer(serializers.Serializer):
    """로그인 시리얼라이저"""
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

