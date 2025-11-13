"""
인증 앱 설정
"""
from django.apps import AppConfig


class AuthConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app.auth'
    label = 'app_auth'
    verbose_name = '인증'

