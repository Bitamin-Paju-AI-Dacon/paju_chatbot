"""
챗봇 앱 설정
"""
from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app.chat'
    verbose_name = '챗봇'

