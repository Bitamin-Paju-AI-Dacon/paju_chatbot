"""
보상 시스템 Serializer
"""
from rest_framework import serializers
from app.chat.models import Stamp, Reward, ClaimedReward


class StampSerializer(serializers.ModelSerializer):
    """스탬프 Serializer"""
    
    class Meta:
        model = Stamp
        fields = ['id', 'place', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class RewardSerializer(serializers.ModelSerializer):
    """보상 Serializer"""
    can_claim = serializers.SerializerMethodField()
    
    class Meta:
        model = Reward
        fields = ['id', 'name', 'type', 'required_stamps', 'expiry_days', 'can_claim']
        read_only_fields = ['id']
    
    def get_can_claim(self, obj):
        """현재 사용자가 이 보상을 받을 수 있는지 여부"""
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return False
        
        user = request.user
        total_stamps = Stamp.objects.filter(user=user).count()
        return total_stamps >= obj.required_stamps


class ClaimedRewardSerializer(serializers.ModelSerializer):
    """받은 보상 Serializer"""
    reward_id = serializers.IntegerField(source='reward.id', read_only=True)
    reward_name = serializers.CharField(source='reward.name', read_only=True)
    reward_type = serializers.CharField(source='reward.type', read_only=True)
    
    class Meta:
        model = ClaimedReward
        fields = [
            'id', 'reward_id', 'reward_name', 'reward_type',
            'claimed_date', 'expiry_date', 'status', 'code'
        ]
        read_only_fields = ['id', 'claimed_date', 'expiry_date', 'status']


class ClaimRewardSerializer(serializers.Serializer):
    """보상 받기 요청 Serializer"""
    reward_id = serializers.IntegerField()

