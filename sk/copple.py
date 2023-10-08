import asyncio
import boto3
import json
import uuid
from datetime import datetime
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from boto3.dynamodb.conditions import Key

# DynamoDB 연결
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Account')
# 대화내용을 모두 저장하는 테이블 연결
conversation_history_table = dynamodb.Table('Chat')

# Query를 사용하여 특정 조건을 만족하는 아이템 가져오기
response = table.query(
    KeyConditionExpression=Key('UserId').eq('Copple') & Key('UserName').eq('코플')
)

# 결과에서 UserId만 추출
user_info = {item.get('UserId') for item in response.get('Items', [])}

print(user_info)

system_message = """
너는 챗봇이야. 사람들의 질문을 받고 너가 알맞게 답변을 해주는 챗봇이야. Based on DynamoDB Table Information, you know the usres' info.
"""

kernel = sk.Kernel()

# api_key 연결
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)

prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=500, temperature=0.7, top_p=0.8
)

prompt_template = sk.ChatPromptTemplate(
    "{{$user_input}}", kernel.prompt_template_engine, prompt_config
)

prompt_template.add_system_message(system_message)
prompt_template.add_user_message("Hi there, who are you?")
prompt_template.add_assistant_message("저는 코플 챗봇입니다. 어떤게 필요하신가요?")

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)


async def chat() -> bool:
    context_vars = sk.ContextVariables()

    try:
        user_input = input("User:> ")
        context_vars["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    # 대화내용 가져오기
    conversation_history = list(context_vars.get("conversation_history") or [])

    # 사용자 입력을 대화 기록에 추가
    conversation_history.append({'role': 'user', 'content': user_input})
    context_vars["conversation_history"] = conversation_history

    # 챗봇 함수 실행
    answer = await kernel.run_async(chat_function, input_vars=context_vars)

    # 챗봇 응답을 대화 기록에 추가
    conversation_history.append({'role': 'assistant', 'content': answer})
    context_vars['conversation_history'] = conversation_history

    # 대화기록을 DynamoDB에 저장
    conversation_id = str(uuid.uuid4())  # 각 대화에 대한 고유 ID를 생성
    created_time = datetime.utcnow().isoformat() + "Z"

    # DynamoDB에 저장할 아이템 구성
    user_item = {
        "Id": str(uuid.uuid4()),
        "ParentId": 'copple',
        "CreatedTime": created_time,
        "Message": user_input,
        "Role": "User"
    }

    assistant_item = {
        "Id": str(uuid.uuid4()),
        "ParentId": 'copple',
        "CreatedTime": created_time,
        "Message": answer,
        "Role": "Assistant"
    }

    # DynamoDB에 사용자와 챗봇 아이템 저장
    conversation_history_table.put_item(Item=user_item)
    conversation_history_table.put_item(Item=assistant_item)

    print(f"Copple:> {answer}")
    return True

async def main() -> None:
    chatting = True
    while chatting:
        chatting = await chat()

if __name__ == "__main__":
    asyncio.run(main())
