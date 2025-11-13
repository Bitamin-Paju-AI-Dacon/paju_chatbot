"""
챗봇 Admin 설정
"""
from django.contrib import admin
from app.chat.models import Stamp, Reward, ClaimedReward, UserImage


@admin.register(Stamp)
class StampAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'place', 'timestamp', 'created_at']
    list_filter = ['timestamp', 'created_at']
    search_fields = ['user__username', 'place']
    readonly_fields = ['timestamp', 'created_at']


@admin.register(Reward)
class RewardAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'type', 'required_stamps', 'expiry_days', 'is_active', 'created_at']
    list_filter = ['type', 'is_active', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at']


@admin.register(ClaimedReward)
class ClaimedRewardAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'reward', 'claimed_date', 'expiry_date', 'status', 'code']
    list_filter = ['status', 'claimed_date', 'expiry_date']
    search_fields = ['user__username', 'reward__name', 'code']
    readonly_fields = ['claimed_date', 'created_at']


@admin.register(UserImage)
class UserImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'place_name', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['user__username', 'place_name']
    readonly_fields = ['uploaded_at']

