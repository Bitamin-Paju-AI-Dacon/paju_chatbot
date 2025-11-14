#!/bin/bash
set -xe

DOCKER_IMAGE=$1

echo "app 폴더 생성 확인"
mkdir -p /home/azureuser/app

echo "[1] .env 파일 생성"
printf "%s\n%s\n" "$ENV_AZURE" "$ENV_DB" > /home/azureuser/app/.env

echo "[2] Docker 이미지 Pull"
docker pull $DOCKER_IMAGE

echo "[3] 기존 컨테이너 정리"
docker stop paju_web || true
docker rm paju_web || true

echo "[4] 새 컨테이너 실행"
docker run -d \
  --name paju_web \
  --env-file /home/azureuser/app/.env \
  -p 8000:8000 \
  -v /home/azureuser/app/media:/app/media \
  -v /home/azureuser/app/staticfiles:/app/staticfiles \
  $DOCKER_IMAGE

echo "[5] 배포 완료!"