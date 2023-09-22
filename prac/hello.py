import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import boto3

# OPENAI 설정 (API키를 .env 파일에서 가져오기)
api_key, org_id = sk.openai_settings_from_dot_env()

# AWS 리소스를 자동으로 인식합니다(환경 변수를 사용)
dynamodb = boto3.resource("dynamodb")

# Semantic Kernel 초기화
kernel = sk.Kernel()

# OpenAIChatCompletion 서비스 추가
kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)

# 사용자 정보를 DynamoDB에서 가져옴
user_id = "test"  # 사용자의 고유 ID

# DynamoDB 테이블 이름 설정
dynamodb_table_name = "Account"

# DynamoDB 테이블 연결
table = dynamodb.Table(dynamodb_table_name)

# DynamoDB에서 id가 "test"인 항목 조회
response = table.get_item(Key={"UserId": user_id})
item = response.get("Item", None)

if item:
    user_name = item.get("user_id", "Unknown")
    # 환영메시지 생성
    welcome_message = f"{user_name}님 안녕하세요. 무엇을 도와드릴까요?"
    print(welcome_message)
else:
    print("사용자를 찾을 수 없습니다.")

while True:
    user_input = input("사용자: ")
    if user_input.lower() == "종료":
        break

        # Semantic Kernel을 사용하여 응답을 생성하기
        response = kernel.process_message("chat-gpt", user_input)
        assistant_response = response.get("message", "죄송해요, 이해할 수 없어요")
        print(f"Assistant: {assistant_response}")
