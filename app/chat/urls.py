"""
챗봇 URL 설정
"""
from django.urls import path
from app.chat.views import (
    greeting_view,
    text_chat_view,
    image_chat_view,
    stamps_view,
    clear_session_view,
    user_images_view,
    delete_user_image_view
)
from app.chat.rewards_views import (
    user_stamps_view,
    available_rewards_view,
    claim_reward_view,
    claimed_rewards_view
)

urlpatterns = [
    path('greeting/', greeting_view, name='greeting'),
    path('text/', text_chat_view, name='text_chat'),
    path('image/', image_chat_view, name='image_chat'),
    path('stamps/<str:session_id>/', stamps_view, name='stamps'),
    path('session/<str:session_id>/', clear_session_view, name='clear_session'),
    # 보상 시스템 API
    path('rewards/stamps/', user_stamps_view, name='user_stamps'),
    path('rewards/available/', available_rewards_view, name='available_rewards'),
    path('rewards/claim/', claim_reward_view, name='claim_reward'),
    path('rewards/claimed/', claimed_rewards_view, name='claimed_rewards'),
    # 사용자 이미지 API
    path('images/', user_images_view, name='user_images'),
    path('images/<int:image_id>/', delete_user_image_view, name='delete_user_image'),
]

