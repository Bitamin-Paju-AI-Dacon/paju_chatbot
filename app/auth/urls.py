"""
인증 URL 설정
"""
from django.urls import path
from app.auth.views import SignupView, login_view, me_view

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', login_view, name='login'),
    path('me/', me_view, name='me'),
]

