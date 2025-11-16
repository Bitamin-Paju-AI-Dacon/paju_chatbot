import os, json, mimetypes, torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from openai import AzureOpenAI
from dotenv import load_dotenv
from retriever import retrieve_event_info

load_dotenv()

def get_client():
    """매 요청마다 환경변수 기반으로 AzureOpenAI 클라이언트를 생성"""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )


# 이미지 분류 모델 설정 불러오기
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

num_classes = cfg["num_classes"]
model_path = cfg["model_path"]
class_names = cfg["class_names"]


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)



mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

SYSTEM_PROMPT = (
    "너는 파주 출판단지를 안내하는 전문 챗봇이야. "
    "구어체나 감탄사 없이, 안내문 형식의 문어체로 작성해."
)

# 사용자별 대화 히스토리/스탬프 저장소
conversation_sessions = {}
user_stamps = {} 

# GPT 대화 기능
def ask_gpt(user_prompt: str, session_id: str):
    client = get_client()
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation_sessions[session_id].append({"role": "user", "content": user_prompt})

    res = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=conversation_sessions[session_id]
    )

    answer = res.choices[0].message.content
    conversation_sessions[session_id].append({"role": "assistant", "content": answer})
    return answer



IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image_input(x):
    """입력이 이미지인지 판별"""
    if isinstance(x, Image.Image):
        return True
    if isinstance(x, str) and os.path.exists(x):
        ext = os.path.splitext(x)[1].lower()
        mime, _ = mimetypes.guess_type(x)
        return (ext in IMG_EXTS) or ((mime or "").startswith("image/"))
    return False

def predict_place(image_path):
    """이미지 → 건물 분류"""
    try:
        img = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError):
        return "이미지 로드를 실패했습니다. 다시 업로드 해주세요."
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()
    return class_names[pred_idx]


# 행사 RAG 
def text_mode(user_text: str, session_id: str) -> str:
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


# 이미지 모드
def image_mode(image_path: str, session_id: str, user_text: str = None):
    """
    이미지를 분석하여 장소를 예측하고 설명을 제공합니다.
    
    Args:
        image_path: 이미지 경로
        session_id: 사용자 세션 ID
        user_text: 사용자가 함께 입력한 텍스트 (선택사항)
    """
    place_name = predict_place(image_path)
    
    # 사용자가 장소 확인을 원하는 경우 (텍스트가 있거나, 장소 확인 키워드가 포함된 경우)
    place_inquiry_keywords = ["장소", "어디", "뭐야", "무엇", "이곳", "여기", "알려", "설명"]
    wants_stamp = False
    
    if user_text:
        user_text_lower = user_text.lower()
        # 스탬프 관련 키워드가 있으면 스탬프 적립 의도로 판단
        stamp_keywords = ["스탬프", "적립", "체크인"]
        if any(kw in user_text_lower for kw in stamp_keywords):
            wants_stamp = True
        # 장소 확인 의도가 명확한 경우
        elif any(kw in user_text_lower for kw in place_inquiry_keywords):
            wants_stamp = False
    else:
        # 텍스트가 없으면 기본적으로 장소 설명 제공
        wants_stamp = False
    
    # 장소 설명 생성
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
    
    # 스탬프 적립 처리
    stamp_message = ""
    if wants_stamp:
        if session_id not in user_stamps:
            user_stamps[session_id] = []

        if place_name not in user_stamps[session_id]:
            user_stamps[session_id].append(place_name)
            stamp_message = f"\n\n'{place_name}'의 스탬프가 적립되었습니다! 🎉"
        else:
            stamp_message = f"\n\n'{place_name}'은(는) 이미 적립된 장소입니다. 😉"
    
    return {"answer": answer + stamp_message, "label": place_name}



def infer_chat(x, session_id: str, user_text: str = None):
    """
    x: 텍스트(str) or 이미지 경로(str) or PIL.Image
    session_id: 사용자별 고유 ID (예: user_id, 채팅방 id 등)
    user_text: 사용자가 입력한 텍스트 (이미지와 함께 전달된 경우)
    """
    if is_image_input(x):
        return image_mode(x, session_id, user_text)
    else:
        # 텍스트 입력에서 이미지 관련 질문이 있는지 확인
        text = str(x) if user_text is None else user_text
        return {"answer": text_mode(text, session_id), "label": None}


# 인삿말 함수
def get_greeting():
    """앱 첫 실행 시 보여줄 인삿말"""
    greeting = (
        "안녕하세요, 파주 출판단지 챗봇 파랑이입니다.\n"
        "텍스트 입력이나 이미지 업로드를 통해 원하시는 장소의 정보를 안내받을 수 있습니다.\n"
        "또한 출판단지에서 예정된 다양한 행사 일정도 함께 확인하실 수 있습니다.\n\n"
        "사진 업로드 방법:\n"
        "- 사진만 업로드: 장소 정보를 자동으로 안내해드립니다.\n"
        "- 사진과 함께 질문: '이 장소가 뭐야?', '이곳은 어디인가요?' 등으로 질문하시면 더 자세한 정보를 제공합니다.\n"
        "- 스탬프 적립: '스탬프 적립해줘', '체크인' 등과 함께 사진을 업로드하시면 스탬프가 적립됩니다."
    )
    return greeting


if __name__ == "__main__":
    print(get_greeting(), "\n")
    session = "user_001"
    while True:
        sample = input("입력 (quit 입력 시 종료) >> ").strip()
        if sample.lower() == "quit":
            break
        result = infer_chat(sample, session)
        print(f"\n{result['answer']}\n")