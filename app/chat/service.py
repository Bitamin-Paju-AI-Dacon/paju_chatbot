"""
챗봇 서비스 로직
"""
import io
import os
import torch
from PIL import Image
from app.chat.models import client, model, transform, class_names, SYSTEM_PROMPT
from retriever import retrieve_event_info

# 사용자별 대화 히스토리/스탬프 저장소
conversation_sessions = {}
user_stamps = {}


def get_conversation_history(session_id: str):
    """세션별 대화 히스토리 가져오기"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return conversation_sessions[session_id]


def ask_gpt(user_prompt: str, session_id: str) -> str:
    """GPT에게 질문"""
    conversation_history = get_conversation_history(session_id)
    conversation_history.append({"role": "user", "content": user_prompt})
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=conversation_history
    )
    
    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    return answer


def predict_place(image_bytes: bytes):
    """이미지에서 장소 예측"""
    import logging
    logger = logging.getLogger(__name__)
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0][pred_idx].item()
        
        # 디버깅: 모든 클래스의 확률 출력
        probs_list = probabilities[0].tolist()
        logger.info(f"예측 인덱스: {pred_idx}, 예측 클래스: {class_names[pred_idx]}, 신뢰도: {confidence:.4f}")
        logger.info(f"상위 3개 예측:")
        top3_indices = torch.topk(probabilities[0], 3).indices.tolist()
        for idx in top3_indices:
            logger.info(f"  - {class_names[idx]}: {probabilities[0][idx].item():.4f}")
    
    return class_names[pred_idx], confidence


def text_chat(user_text: str, session_id: str) -> str:
    """텍스트 기반 챗봇 대화"""
    # '행사' 또는 '이벤트' 키워드가 들어가면 RAG 검색
    if any(k in user_text for k in ["행사", "이벤트"]):
        results = retrieve_event_info(user_text, top_k=2)
        if results:
            context = "\n\n".join([r.page_content for r in results])
            prompt = f"""
사용자가 '{user_text}'라고 물었어.
아래는 관련 행사 정보야:
{context}

제목, 일시, 장소, 주최, 요약, 신청방법을 보기 좋게 정리해줘.
꼭 사용자가 보기 깔끔하게 출력해줘
"""
            return ask_gpt(prompt, session_id)
        else:
            return "현재 해당 주제의 행사 정보는 없습니다."
    else:
        prompt = f"""
사용자가 '{user_text}'라고 물었어.
파주 출판단지 관련이면 2~3줄 요약 후,
'다른 정보에 대해 궁금하다면 추가로 질문해주세요.'라고 유도 질문을 추가하면서 마무리.
아니면 '죄송하지만 저는 파주 출판단지 관련 정보만 안내할 수 있습니다.'라고만 출력.
"""
        return ask_gpt(prompt, session_id)


def image_chat_with_stamp(image_bytes: bytes, session_id: str, action: str = "description", user_text: str = None):
    """
    이미지 기반 장소 인식 및 처리
    
    Args:
        image_bytes: 이미지 바이트 데이터
        session_id: 세션 ID
        action: "stamp" (스탬프 적립) 또는 "description" (장소 설명)
        user_text: 사용자가 함께 입력한 텍스트 (선택사항)
    
    Returns:
        dict: {"answer": str, "label": str, "confidence": float, "stamp_added": bool}
    """
    # 장소 예측
    place_name, confidence = predict_place(image_bytes)
    
    # 사용자 텍스트가 있으면 의도 파악
    place_inquiry_keywords = ["장소", "어디", "뭐야", "무엇", "이곳", "여기", "알려", "설명"]
    stamp_keywords = ["스탬프", "적립", "체크인"]
    
    # action이 명시되지 않았고 user_text가 있으면 의도 파악
    if user_text:
        user_text_lower = user_text.lower()
        if any(kw in user_text_lower for kw in stamp_keywords):
            action = "stamp"
        elif any(kw in user_text_lower for kw in place_inquiry_keywords):
            action = "description"
    
    # 스탬프 적립
    if action == "stamp":
        if session_id not in user_stamps:
            user_stamps[session_id] = []
        
        if place_name not in user_stamps[session_id]:
            user_stamps[session_id].append(place_name)
            message = f"'{place_name}'의 스탬프가 적립되었습니다! 🎉"
            stamp_added = True
        else:
            message = f"'{place_name}'은(는) 이미 적립된 장소입니다. 😉"
            stamp_added = False
        
        return {
            "answer": message,
            "label": place_name,
            "predicted_place": place_name,  # 프론트엔드 호환성을 위해 추가
            "confidence": round(confidence * 100, 2),
            "stamp_added": stamp_added
        }
    
    # 장소 설명
    else:
        prompt = f"""
사용자가 '{place_name}' 사진을 보냈어.
"""
        if user_text:
            prompt += f"사용자가 '{user_text}'라고 물었어.\n"
        
        prompt += f"""
'{place_name}'이 파주 출판단지 관련이면 2~3줄로 요약하고,
'다른 정보에 대해 궁금하다면 추가로 질문해주세요.'라고 유도 질문을 추가하면서 마무리.
아니면 '죄송하지만 저는 파주 출판단지 관련 정보만 안내할 수 있습니다.'라고만 출력.
"""
        answer = ask_gpt(prompt, session_id)
        return {
            "answer": answer,
            "description": answer,  # 프론트엔드 호환성을 위해 추가
            "label": place_name,
            "predicted_place": place_name,  # 프론트엔드 호환성을 위해 추가
            "confidence": round(confidence * 100, 2),
            "stamp_added": False
        }


def get_user_stamps(session_id: str) -> list:
    """사용자의 스탬프 목록 조회"""
    return user_stamps.get(session_id, [])


def clear_session(session_id: str):
    """세션 대화 히스토리 및 스탬프 초기화"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    if session_id in user_stamps:
        del user_stamps[session_id]


def get_greeting() -> str:
    """앱 첫 실행 시 보여줄 인삿말"""
    return (
        "안녕하세요, 파주 출판단지 챗봇 파랑이입니다.\n"
        "텍스트 입력이나 이미지 업로드를 통해 원하시는 장소의 정보를 안내받을 수 있습니다.\n"
        "또한 출판단지에서 예정된 다양한 행사 일정도 함께 확인하실 수 있습니다.\n"
        "사진을 업로드를 통해 스탬프를 적립할 수도 있어요!"
    )

