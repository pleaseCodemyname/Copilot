import asyncio
import boto3
import uuid
import openai
import re
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

get_prompt = """
You are a classifier that categorizes the input as either a goal, an event, or a to-do:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
Todo: Refers to a small task or duty that needs to be accomplished.
When answering, speak Naturally and when you reply natu either a goal, event or todo.
Categorize the following input as either a goal, event, or todo.
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

crud_prompt  = """
You are an action type recognizer that categorizes the input as either a create, read, update, or delete:
Create: Includes the act of meeting someone or doing something.
Read: Refers to the act of consuming information or data.
Update: Involves modifying or changing existing information or data.
Delete: Contains the meaning of deleting or making something disappear, Eradication, Elimination.
When answering, please answer the type of action and Say it in a soft tone
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

extract_prompt = """
You need to know how to distinguish parameters based on the values you enter. First, the default parameter is "title", "startDatime", "endDatime", "location", "content", "image" and "isCompleted".
title should contain something symbolic or representative of the value entered by the user. 
startDatetime is An event, schedule, or schedule begins. Must specify "year-month-day-hour-minute" with no words. only datetime.
endDatime is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "year-month-day-hour-minutes" with no words. only datetime.
location means meeting someone or a place. 
I hope the content includes anything other than behavioral and planning. 
image value is always null.
isCompleted value begins with false unless users specify when they say or mention it is done or complete. They they say or mention it is complete, change the value from false to true.
When answering, ignore special characters, symbols and blank spaces. Just answer title, startDatetime, endDatetime, location,content, image, and isCompleted.
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
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

crud_template = sk.PromptTemplate(
    crud_prompt, kernel.prompt_template_engine, prompt_config
)

get_template = sk.PromptTemplate(
    get_prompt, kernel.prompt_template_engine, prompt_config
)

extract_template = sk.PromptTemplate(
    extract_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

get_config = sk.SemanticFunctionConfig(prompt_config, get_template)
get_function = kernel.register_semantic_function("GET", "Get", get_config)

crud_config = sk.SemanticFunctionConfig(prompt_config, crud_template)
crud_function = kernel.register_semantic_function("CRUD", "Crud", crud_config)

extract_config = sk.SemanticFunctionConfig(prompt_config, extract_template)
extract_function = kernel.register_semantic_function("EXTRACT", "Extract", extract_config)

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

# Goal, Event, Todo 중 하나를 인식하는 함수
async def recognize_get_intent(user_input, context_vars):
    try:
        # 사용자 입력과 함께 get_prompt 실행
        answer = await kernel.run_async(get_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})
        
        # Goal, Event, Todo 중 어떤 키워드가 나왔는지 확인
        category = None
        if "goal" in answer.lower():
            category = "Goal"
        elif "event" in answer.lower():
            category = "Event"
        elif "todo" in answer.lower():
            category = "Todo"
        
        # 어떤 카테고리인지와 함께 답변 내용도 반환
        return category, answer.choices[0].text.strip()
    except Exception as e:
        return None, None

# 사용자의 CRUD 의도 추출 함수
async def recognize_crud_intent(user_input, context_vars):
    crud_keywords = {
        "create": ["약속", "만나기", "계획", "추가"],
        "read": ["읽기", "확인", "보기"],
        "update": ["수정", "갱신", "변경"],
        "delete": ["삭제", "지우기", "제거"]
    }

    for crud, keywords in crud_keywords.items():
        for keyword in keywords:
            if keyword in user_input.lower():
                return crud

    return None

# 사용자의 대화내용에 맞게 Data 형식 갖추기
async def extract_data(user_input, context_vars):
    answer = await kernel.run_async(extract_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})

    # 여기에 데이터 활용 로직 추가
    extracted_data = {}  # 추가: 데이터를 저장할 딕셔너리 생성
    split_answer_text = answer.choices[0].text.strip().split("\n")  # 추가: 답변 텍스트를 줄 단위로 분할

    if "create" in context_vars['user_input'].lower():
        # 'create' 동작에 대한 데이터 처리
        for line in split_answer_text:
            if "title" in line:
                extracted_data["title"] = line.split(":")[1].strip()
            elif "startDatetime" in line:
                extracted_data["startDatetime"] = line.split(":")[1].strip()
            elif "endDatetime" in line:
                extracted_data["endDatetime"] = line.split(":")[1].strip()
            elif "location" in line:
                extracted_data["location"] = line.split(":")[1].strip()
            elif "content" in line:
                extracted_data["content"] = line.split(":")[1].strip()
            elif "image" in line:
                extracted_data["image"] = line.split(":")[1].strip()
            elif "isCompleted" in line:
                extracted_data["isCompleted"] = line.split(":")[1].strip()

        print("Processing 'create' data...")
        print("Extracted Data:", extracted_data)

    elif "read" in context_vars['user_input'].lower():
        # 'read' 동작에 대한 데이터 처리
        print("Processing 'read' data...")

    elif "update" in context_vars['user_input'].lower():
        # 'update' 동작에 대한 데이터 처리
        print("Processing 'update' data...")

    elif "delete" in context_vars['user_input'].lower():
        # 'delete' 동작에 대한 데이터 처리
        print("Processing 'delete' data...")
# Chat 함수 내부 수정
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

    if user_input.lower() == "exit":
        print("\n\nExiting chat...")
        return False

    # Get 의도 인식
    get_intent, answer_text = await recognize_get_intent(user_input, context_vars)
    print(get_intent)

    # Get 의도가 감지되면 CRUD 의도 판별
    if get_intent:
        crud_intent = await recognize_crud_intent(user_input, context_vars)

        if not crud_intent:
            # 명시적인 CRUD 동사 추출
            crud_intent = await recognize_crud_intent(user_input, context_vars)

        if crud_intent:
            # Get 및 CRUD 의도가 감지되면 각각에 해당하는 함수 실행
            if get_intent:
                context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {get_intent}\n"
                context_vars["chat_history"] += f"ChatBot:> {answer_text}\n"

            if crud_intent:
                answer = await kernel.run_async(crud_function, input_vars=context_vars)
                context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> CRUD Intent Recognized: {crud_intent}\n"
                context_vars["chat_history"] += f"ChatBot:> {answer}\n"

                # CRUD 동작에 따라 데이터 처리
                await extract_data(user_input, context_vars)
        else:
            # Get이 감지되었지만 CRUD 의도가 없으면 일반 챗봇 기능 수행
            answer = await kernel.run_async(chat_function, input_vars=context_vars)
            context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"
    else:
        # Get이 감지되지 않으면 일반 챗봇 기능 수행
        answer = await kernel.run_async(chat_function, input_vars=context_vars)
        context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"

    print(f"ChatBot:> {answer}")
    return True  # 여기에서는 대화가 종료된 후에만 저장하도록 변경

# main 함수 수정
async def main() -> None:
    context = sk.ContextVariables()
    context["chat_history"] = ""

    chatting = True
    while chatting:
        chatting = await chat(context)

        # 대화가 종료된 후에 한 번만 최종 기록을 저장
        if not chatting:
            try:
                save_to_dynamodb(context)
            except Exception as e:
                print(f"Error saving to DynamoDB: {e}")

if __name__ == "__main__":
    asyncio.run(main())
