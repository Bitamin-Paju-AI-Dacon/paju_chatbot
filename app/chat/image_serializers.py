"""
사용자 이미지 Serializer
"""
from rest_framework import serializers
from app.chat.models import UserImage


class UserImageSerializer(serializers.ModelSerializer):
    """사용자 이미지 Serializer"""
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = UserImage
        fields = ['id', 'image_url', 'place_name', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']
    
    def get_image_url(self, obj):
        """이미지 URL 반환"""
        request = self.context.get('request')
        if obj.image and hasattr(obj.image, 'url'):
            if request:
                return request.build_absolute_uri(obj.image.url)
            return obj.image.url
        return None

