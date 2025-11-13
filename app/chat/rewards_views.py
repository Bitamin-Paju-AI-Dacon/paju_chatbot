"""
보상 시스템 API 뷰
"""
import uuid
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta
from app.chat.models import Stamp, Reward, ClaimedReward
from app.chat.rewards_serializers import (
    StampSerializer,
    RewardSerializer,
    ClaimedRewardSerializer,
    ClaimRewardSerializer
)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_stamps_view(request):
    """사용자별 스탬프 개수 조회"""
    user = request.user
    stamps = Stamp.objects.filter(user=user).order_by('-timestamp')
    
    serializer = StampSerializer(stamps, many=True)
    
    return Response({
        "total_stamps": stamps.count(),
        "stamps": serializer.data
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def available_rewards_view(request):
    """받을 수 있는 보상 목록 조회"""
    user = request.user
    total_stamps = Stamp.objects.filter(user=user).count()
    
    # 활성화된 보상만 조회
    rewards = Reward.objects.filter(is_active=True).order_by('required_stamps')
    
    serializer = RewardSerializer(rewards, many=True, context={'request': request})
    
    return Response({
        "available_rewards": serializer.data,
        "total_stamps": total_stamps
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def claim_reward_view(request):
    """보상 받기"""
    user = request.user
    serializer = ClaimRewardSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    
    reward_id = serializer.validated_data['reward_id']
    
    try:
        reward = Reward.objects.get(id=reward_id, is_active=True)
    except Reward.DoesNotExist:
        return Response(
            {"detail": "보상을 찾을 수 없습니다."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # 이미 받은 보상인지 확인
    existing_claim = ClaimedReward.objects.filter(
        user=user,
        reward=reward,
        status='사용 가능'
    ).first()
    
    if existing_claim:
        # 만료되지 않았는지 확인
        if timezone.now() < existing_claim.expiry_date:
            return Response(
                {"detail": "이미 받은 보상입니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # 스탬프 개수 확인
    total_stamps = Stamp.objects.filter(user=user).count()
    if total_stamps < reward.required_stamps:
        return Response(
            {
                "detail": f"스탬프가 부족합니다. (현재: {total_stamps}개, 필요: {reward.required_stamps}개)"
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # 쿠폰 코드 생성 (쿠폰 타입인 경우)
    code = None
    if reward.type == '쿠폰':
        # 간단한 쿠폰 코드 생성 (예: REWARD-{reward_id}-{uuid})
        code = f"REWARD-{reward_id}-{str(uuid.uuid4())[:8].upper()}"
    
    # 보상 받기
    expiry_date = timezone.now() + timedelta(days=reward.expiry_days)
    claimed_reward = ClaimedReward.objects.create(
        user=user,
        reward=reward,
        expiry_date=expiry_date,
        code=code,
        status='사용 가능'
    )
    
    serializer = ClaimedRewardSerializer(claimed_reward)
    
    return Response({
        "success": True,
        "reward": serializer.data,
        "message": "보상을 성공적으로 받았습니다."
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def claimed_rewards_view(request):
    """받은 보상 목록 조회"""
    user = request.user
    claimed_rewards = ClaimedReward.objects.filter(user=user).order_by('-claimed_date')
    
    # 만료 상태 자동 업데이트
    for claimed_reward in claimed_rewards:
        if claimed_reward.status == '사용 가능' and timezone.now() > claimed_reward.expiry_date:
            claimed_reward.status = '만료됨'
            claimed_reward.save()
    
    serializer = ClaimedRewardSerializer(claimed_rewards, many=True)
    
    return Response({
        "claimed_rewards": serializer.data,
        "count": claimed_rewards.count()
    })

