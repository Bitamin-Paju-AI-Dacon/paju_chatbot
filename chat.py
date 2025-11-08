import os, json
import torch
from torchvision import models, transforms
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv
from retriever import retrieve_event_info  


load_dotenv()

client = AzureOpenAI(
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
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 정규화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_place(image_path):
    """이미지 → 장소 예측"""
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
    return class_names[pred_idx]

# GPT 대화 기능
conversation_history = [
    {"role": "system", "content": "너는 파주 출판단지를 안내하는 전문 챗봇이야. 구어체나 감탄사 없이, 안내문 형식의 문어체로 작성해"}
]

def ask_gpt(prompt):
    conversation_history.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=conversation_history
    )
    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    return answer


# 텍스트 모드 (행사 RAG 연동)
def text_mode():
    while True:
        user_input = input("사용자: ").strip()
        if user_input.lower() == "quit":
            print("챗봇: 이용해 주셔서 감사합니다.")
            break

        # '행사' 또는 '이벤트' 키워드가 들어가면 RAG 검색
        if any(keyword in user_input for keyword in ["행사", "이벤트"]):
            results = retrieve_event_info(user_input, top_k=2)
            if results:
                context = "\n\n".join([r.page_content for r in results])
                prompt = f"""
                사용자가 '{user_input}'라고 물었어.
                아래는 관련된 행사 정보야:
                {context}

                위 내용을 참고하여 사용자의 질문에 구체적이고 자연스럽게 답변해.
                제목, 일시, 장소, 호스트, 핵심 요약, 신청방법 및 신청 링크 위주로 보기 좋게 정리하고,
                불필요한 문장은 생략해.
                """
                answer = ask_gpt(prompt)
                print(f"\n📅 행사 정보\n{answer}\n")
                continue
            else:
                print("챗봇: 현재 해당 주제의 행사 정보는 없습니다.\n")
                continue

        # 행사 외 질문
        prompt = f"""
        사용자가 '{user_input}'라고 물었어.

        1. 파주 출판단지 관련 질문이면:
           - 장소의 핵심 요약만 2~3줄로 알려줘.
           - 마지막에 '다른 정보에 대해 궁금하다면 추가로 질문해주세요'라고 유도 질문을 덧붙여.

        2. 관련 없는 질문이면:
           - "죄송하지만 저는 파주 출판단지 관련 정보만 안내할 수 있습니다." 라고만 출력.
        """
        answer = ask_gpt(prompt)
        print(f"\n챗봇: {answer}\n")

        # 추가 대화 유도
        follow_up = input("사용자: ").strip().lower()
        if follow_up in ["응", "좋아요", "ㅇㅋ", "더 알려줘", "그래"]:
            detail_prompt = f"'{user_input}'에 대해 자세히 안내문 형식으로 써줘."
            detail_answer = ask_gpt(detail_prompt)
            print(f"\n챗봇: {detail_answer}\n")
        elif follow_up in ["아니", "괜찮아요", "그만"]:
            print("챗봇: 알겠습니다. 다른 장소나 궁금한 점이 있나요?\n")
        else:
            next_answer = ask_gpt(f"사용자가 '{follow_up}'라고 대답했어. 자연스럽게 대화를 이어가줘.")
            print(f"\n챗봇: {next_answer}\n")


# 이미지 모드
def image_mode():
    while True:
        image_path = input("이미지 파일 경로 입력 (종료하려면 quit): ").strip()
        if image_path.lower() == "quit":
            print("챗봇: 이미지 모드를 종료합니다.")
            break

        if not os.path.exists(image_path):
            print("⚠️ 파일이 존재하지 않습니다. 다시 입력해주세요.\n")
            continue

        place_name = predict_place(image_path)
        print(f"\n[모델 예측 장소] {place_name}\n")

        prompt = f"""
        사용자가 '{place_name}' 사진을 보냈어.
        이 장소가 파주 출판단지와 관련이 있다면:
        - '{place_name}'의 핵심 특징을 2~3줄로 요약하고,
        - 마지막에 '다른 정보에 대해 궁금하다면 추가로 질문해주세요'라고 유도 질문을 추가해.
        관련이 없다면 "죄송하지만 저는 파주 출판단지 관련 정보만 안내할 수 있습니다."라고만 출력해.
        """
        answer = ask_gpt(prompt)
        print(f"\n챗봇: {answer}\n")

        follow_up = input("사용자: ").strip().lower()
        if follow_up in ["응", "좋아요", "ㅇㅋ", "더 알려줘", "그래"]:
            detail_prompt = f"'{place_name}'에 대해 자세한 설명(분위기, 방문 포인트, 참고사항)을 안내문 형식으로 써줘."
            detail_answer = ask_gpt(detail_prompt)
            print(f"\n챗봇: {detail_answer}\n")
        elif follow_up in ["아니", "괜찮아요", "그만"]:
            print("챗봇: 알겠습니다. 다른 사진이나 궁금한 점이 있나요?\n")
        else:
            next_answer = ask_gpt(f"사용자가 '{follow_up}'라고 대답했어. 자연스럽게 대화를 이어가줘.")
            print(f"\n챗봇: {next_answer}\n")


def chatbot_interface():
    print("=== 파주 출판단지 안내 챗봇 ===")
    print("안녕하세요, 파주 출판단지 챗봇입니다.")
    print("텍스트 입력이나 이미지 업로드를 통해 원하시는 장소의 정보를 안내받을 수 있습니다.")
    print("또한 출판단지에서 예정된 다양한 행사 일정도 함께 확인하실 수 있습니다.\n")
    print("원하는 모드를 선택해주세요!\n")

    first_run = True

    while True:
        if first_run:
            print("1. 텍스트 질문 (행사 검색 포함)")
            print("2. 이미지 업로드")
            print("종료하려면 'quit' 입력\n")
            first_run = False

        mode = input(">> 모드 선택 (1=text, 2=image): ").strip()
        if mode.lower() == "quit":
            print("챗봇: 프로그램을 종료합니다.")
            break
        elif mode == "1":
            text_mode()
        elif mode == "2":
            image_mode()
        else:
            print("⚠️ 잘못된 입력입니다. 1 또는 2를 선택해주세요.\n")


if __name__ == "__main__":
    chatbot_interface()
