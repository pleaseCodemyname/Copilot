import asyncio
import boto3
import uuid
import semantic_kernel as sk
from botocore.exceptions import NoCredentialsError

import semantic_kernel.connectors.ai.open_ai as sk_oai
from datetime import datetime

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know'
when it doesn't know the answer.
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

intent_prompt = """
You are a classifier that categorizes the input as either a goal, an event, or a to-do:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
To-Do: Refers to a small task or duty that needs to be accomplished.
When answering, speak Naturally and when you reply natu either a goal, event or todo.
"""

kernel = sk.Kernel()
# DynamoDB 연결
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Chat')

#.env파일 연결
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", sk_oai.OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)

prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=2000, temperature=0.7, top_p=0.4
)

prompt_template = sk.PromptTemplate(
    sk_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

async def chat(context_vars: sk.ContextVariables) -> bool:
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

    answer = await kernel.run_async(chat_function, input_vars=context_vars)
    context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"

    save_to_dynamodb(context_vars)

    print(f"ChatBot:> {answer}")
    return True

# DynamoDB에 대화 기록 저장 함수
def save_to_dynamodb(context_vars: sk.ContextVariables) -> None:
    try:
        timestamp = datetime.now().isoformat()
        user_input = context_vars["user_input"]
        chat_history = context_vars["chat_history"]

        while True:
            new_id = str(uuid.uuid4())
            try:
                # DynamoDB에 아이템 추가
                table.put_item(
                    Item={
                        'Id': new_id,  # 새로운 ID 생성
                        'ParentId': "Copple",
                        'CreatedTime': timestamp,
                        'Message': chat_history,
                        'Role': "1"
                    }
                )
                break  # ID가 중복되지 않는 경우 반복문 종료
            except dynamodb.exceptions.ConditionalCheckFailedException:
                pass  # ID가 이미 존재하면 다시 생성

    except NoCredentialsError:
        pass  # AWS credentials 예외 무시

async def main() -> None:
    context = sk.ContextVariables()
    context["chat_history"] = ""

    chatting = True
    while chatting:
        chatting = await chat(context)

if __name__ == "__main__":
    asyncio.run(main())