"""
챗봇 모델 및 설정
"""
import os
import json
import torch
from torchvision import models as torch_models, transforms
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta

load_dotenv()

User = get_user_model()

# OpenAI 클라이언트
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# 이미지 분류 모델 설정
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

num_classes = cfg["num_classes"]
model_path = cfg["model_path"]
class_names = cfg["class_names"]

# ResNet18 모델 로드
model = torch_models.resnet18(weights=torch_models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to('cpu')
model.eval()

# 이미지 전처리
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 시스템 프롬프트
SYSTEM_PROMPT = (
    "너는 파주 출판단지를 안내하는 전문 챗봇이야. "
    "구어체나 감탄사 없이, 안내문 형식의 문어체로 작성해."
)


# ========== Django 모델 ==========

class Stamp(models.Model):
    """스탬프 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='stamps')
    place = models.CharField(max_length=100, db_index=True)
    session_id = models.CharField(max_length=100, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'stamps'
        verbose_name = '스탬프'
        verbose_name_plural = '스탬프들'
        ordering = ['-timestamp']
        # 같은 사용자가 같은 장소에 중복 스탬프를 받지 않도록 (선택사항)
        # unique_together = [['user', 'place']]
    
    def __str__(self):
        return f"{self.user.username} - {self.place}"


class Reward(models.Model):
    """보상 모델"""
    REWARD_TYPES = [
        ('쿠폰', '쿠폰'),
        ('입장권', '입장권'),
        ('응모권', '응모권'),
    ]
    
    name = models.CharField(max_length=200)
    type = models.CharField(max_length=20, choices=REWARD_TYPES)
    required_stamps = models.IntegerField()
    expiry_days = models.IntegerField(default=90)  # 보상 유효 기간 (일)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'rewards'
        verbose_name = '보상'
        verbose_name_plural = '보상들'
        ordering = ['required_stamps']
    
    def __str__(self):
        return f"{self.name} ({self.required_stamps}개 스탬프)"


class ClaimedReward(models.Model):
    """받은 보상 모델"""
    STATUS_CHOICES = [
        ('사용 가능', '사용 가능'),
        ('만료됨', '만료됨'),
        ('사용됨', '사용됨'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='claimed_rewards')
    reward = models.ForeignKey(Reward, on_delete=models.CASCADE, related_name='claims')
    claimed_date = models.DateTimeField(auto_now_add=True)
    expiry_date = models.DateTimeField()
    code = models.CharField(max_length=100, null=True, blank=True)  # 쿠폰 코드
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='사용 가능')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'claimed_rewards'
        verbose_name = '받은 보상'
        verbose_name_plural = '받은 보상들'
        ordering = ['-claimed_date']
    
    def __str__(self):
        return f"{self.user.username} - {self.reward.name}"
    
    def save(self, *args, **kwargs):
        # 만료일 자동 계산
        if not self.expiry_date:
            self.expiry_date = timezone.now() + timedelta(days=self.reward.expiry_days)
        
        # 만료 상태 자동 업데이트
        if self.status == '사용 가능' and timezone.now() > self.expiry_date:
            self.status = '만료됨'
        
        super().save(*args, **kwargs)


class UserImage(models.Model):
    """사용자가 업로드한 이미지 모델"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_images')
    image = models.ImageField(upload_to='user_images/%Y/%m/%d/')
    place_name = models.CharField(max_length=100, null=True, blank=True)  # 예측된 장소 이름
    session_id = models.CharField(max_length=100, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_images'
        verbose_name = '사용자 이미지'
        verbose_name_plural = '사용자 이미지들'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.place_name or 'Unknown'} - {self.uploaded_at}"

