"""
챗봇 API 뷰
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from app.chat.service import (
    text_chat,
    image_chat_with_stamp,
    get_user_stamps,
    clear_session,
    get_greeting
)
from app.chat.models import Stamp, UserImage
from rest_framework.permissions import IsAuthenticated
from app.chat.image_serializers import UserImageSerializer
import os
import logging

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def greeting_view(request):
    """챗봇 인삿말 조회"""
    return Response({"greeting": get_greeting()})


@api_view(['POST'])
@permission_classes([AllowAny])
def text_chat_view(request):
    """텍스트 기반 챗봇 대화"""
    try:
        message = request.data.get('message', '').strip()
        session_id = request.data.get('session_id', 'default')
        
        if not message:
            return Response(
                {"detail": "메시지가 비어있습니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        answer = text_chat(message, session_id)
        return Response({
            "response": answer,
            "session_id": session_id
        })
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def image_chat_view(request):
    """이미지 기반 장소 인식 및 처리"""
    try:
        if 'file' not in request.FILES:
            return Response(
                {"detail": "이미지 파일이 필요합니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        file = request.FILES['file']
        session_id = request.data.get('session_id', 'default')
        action = request.data.get('action', 'description')
        user_text = request.data.get('message', None) or request.data.get('text', None)  # 텍스트 메시지 받기
        
        # 이미지 파일 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            return Response(
                {"detail": "이미지 파일만 업로드 가능합니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # action 검증
        if action not in ['description', 'stamp']:
            return Response(
                {"detail": "action은 'description' 또는 'stamp'여야 합니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 이미지 읽기
        image_bytes = file.read()
        
        # 이미지 처리 (user_text 전달)
        result = image_chat_with_stamp(image_bytes, session_id, action, user_text)
        result['session_id'] = session_id
        
        # 로그인한 사용자인 경우 이미지 저장 (에러가 발생해도 메인 응답은 성공)
        if request.user.is_authenticated:
            try:
                place_name = result.get('label')
                
                # 이미지 저장
                file.seek(0)  # 파일 포인터를 처음으로 되돌림
                user_image = UserImage.objects.create(
                    user=request.user,
                    image=file,
                    place_name=place_name,
                    session_id=session_id
                )
                
                # 이미지 URL 추가
                result['image_url'] = request.build_absolute_uri(user_image.image.url)
                result['image_id'] = user_image.id
                
                # 스탬프를 받은 경우 DB에 저장
                if result.get('stamp_added') and place_name:
                    try:
                        # 중복 체크 (같은 사용자가 같은 장소에 이미 스탬프가 있는지 확인)
                        existing_stamp = Stamp.objects.filter(
                            user=request.user,
                            place=place_name
                        ).first()
                        
                        if not existing_stamp:
                            Stamp.objects.create(
                                user=request.user,
                                place=place_name,
                                session_id=session_id
                            )
                    except Exception as stamp_error:
                        # 스탬프 저장 실패는 무시 (이미 응답은 성공)
                        logger.warning(f"스탬프 저장 실패: {str(stamp_error)}")
                        
            except Exception as image_error:
                # 이미지 저장 실패는 무시 (이미 응답은 성공)
                logger.warning(f"이미지 저장 실패: {str(image_error)}")
                # 이미지 저장 실패해도 메인 응답은 정상 반환
        
        return Response(result)
        
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def stamps_view(request, session_id):
    """사용자의 스탬프 목록 조회"""
    try:
        stamps = get_user_stamps(session_id)
        return Response({
            "stamps": stamps,
            "count": len(stamps),
            "session_id": session_id
        })
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
@permission_classes([AllowAny])
def clear_session_view(request, session_id):
    """세션 대화 히스토리 및 스탬프 초기화"""
    try:
        clear_session(session_id)
        return Response({
            "message": f"Session {session_id} cleared",
            "session_id": session_id
        })
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_images_view(request):
    """사용자가 업로드한 이미지 목록 조회"""
    try:
        images = UserImage.objects.filter(user=request.user).order_by('-uploaded_at')
        serializer = UserImageSerializer(images, many=True, context={'request': request})
        
        return Response({
            "images": serializer.data,
            "count": images.count()
        })
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_user_image_view(request, image_id):
    """사용자 이미지 삭제"""
    try:
        image = UserImage.objects.get(id=image_id, user=request.user)
        
        # 파일 삭제
        if image.image:
            image_path = image.image.path
            if os.path.exists(image_path):
                os.remove(image_path)
        
        image.delete()
        
        return Response({
            "message": "이미지가 삭제되었습니다.",
            "image_id": image_id
        })
    except UserImage.DoesNotExist:
        return Response(
            {"detail": "이미지를 찾을 수 없습니다."},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"detail": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

