#!/bin/bash
set -xe

DOCKER_IMAGE=$1

echo "app 폴더 생성 확인"
mkdir -p /home/azureuser/app

echo "[1] .env 파일 생성"
printf "%s\n%s\n" "$ENV_AZURE" "$ENV_DB" > /home/azureuser/app/.env

echo "[2] Docker 이미지 Pull"
docker pull $DOCKER_IMAGE

# config.json에서 모델 파일명 확인
echo "[3] 모델 파일명 확인"
MODEL_FILENAME=$(docker run --rm $DOCKER_IMAGE cat /app/config.json 2>/dev/null | grep -o '"model_path": "[^"]*"' | cut -d'"' -f4 || echo "paju_model_resnet18_finetuned2.pth")

if [ -z "$MODEL_FILENAME" ]; then
    MODEL_FILENAME="paju_model_resnet18_finetuned2.pth"
fi

echo "모델 파일명: $MODEL_FILENAME"
MODEL_PATH="/home/azureuser/app/$MODEL_FILENAME"

# 모델 파일 존재 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo "   경고: 모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    echo "   모델 파일을 서버에 업로드해주세요:"
    echo "   scp $MODEL_FILENAME azureuser@서버주소:/home/azureuser/app/"
    echo "   배포는 계속 진행하지만 모델이 로드되지 않을 수 있습니다."
else
    echo "모델 파일 확인됨: $MODEL_PATH"
fi

echo "[4] 기존 컨테이너 정리"
docker stop paju_web || true
docker rm paju_web || true

echo "[5] 새 컨테이너 실행"
docker run -d \
  --name paju_web \
  --env-file /home/azureuser/app/.env \
  -p 8000:8000 \
  -v /home/azureuser/app/media:/app/media \
  -v /home/azureuser/app/staticfiles:/app/staticfiles \
  -v "$MODEL_PATH:/app/$MODEL_FILENAME" \
  $DOCKER_IMAGE

echo "[6] 컨테이너 시작 대기"
sleep 10

echo "[7] 컨테이너 내부 모델 파일 확인"
docker exec paju_web ls -lh /app/paju_model_resnet18_finetuned2.pth 2>&1 || echo "컨테이너 내부에서 모델 파일을 찾을 수 없습니다."

echo "[8] 컨테이너 내부 config.json 확인"
docker exec paju_web cat /app/config.json 2>&1 || echo "config.json을 읽을 수 없습니다."

echo "[9] 모델 로드 상태 확인"
echo "=== 전체 로그 (최근 100줄) ==="
docker logs paju_web 2>&1 | tail -100

echo ""
echo "=== 모델 관련 로그 ==="
docker logs paju_web 2>&1 | grep -i -E "(모델|model|config\.json|파일|file|로드|load|error|ERROR|warning|WARNING|성공)" || echo "모델 관련 로그를 찾을 수 없습니다."

echo ""
echo "=== Python에서 모델 직접 테스트 ==="
docker exec paju_web python -c "
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paju_chatbot.settings')
django.setup()

from app.chat import models
print('✓ 모델 로드 완료')
print(f'✓ 모델 타입: {type(models.model)}')
print(f'✓ 모델 상태: {models.model.training}')
print(f'✓ 클래스 개수: {models.num_classes}')
print(f'✓ 클래스 목록: {models.class_names[:3]}...')
" 2>&1 || echo "모델 테스트 실패"

echo ""
echo "=== Django 서버 상태 확인 ==="
docker exec paju_web ps aux | grep -E "(python|manage.py)" || echo "서버 프로세스 확인 실패"

echo "[10] 배포 완료"
