import openai
import asyncio
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
import re

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know'
when it doesn't know the answer.
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

"""

kernel = sk.Kernel()

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

extract_template = sk.PromptTemplate(
    extract_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

extract_function_config = sk.SemanticFunctionConfig(prompt_config, extract_template)
extract_function = kernel.register_semantic_function("EXTRACT", "Extract", extract_function_config)

async def extract_data(user_input, context_vars):
    try:
        # 'chat_history'를 초기화
        if 'chat_history' not in context_vars:
            context_vars['chat_history'] = ""

        # OpenAI와 대화 시도
        response = await kernel.run_async(extract_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})

        # OpenAI의 응답에서 텍스트를 추출하고 앞뒤 공백을 제거합니다.
        answer_text = response.choices[0].text.strip()

        # 추출된 텍스트를 줄 단위로 나눕니다.
        split_answer_text = answer_text.split("\n")

        response_dict = {}  # 딕셔너리를 사용하여 결과를 저장

        # 각 줄을 처리하여 파라미터를 추출합니다.
        for line in split_answer_text:
            # 줄을 콜론(:)을 기준으로 나눕니다.
            parts = line.split(":")

            # 적어도 두 개의 부분으로 나눠진 경우에만 처리합니다.
            if len(parts) >= 2:
                # 첫 번째 부분을 키로, 나머지 부분을 값으로 사용하여 딕셔너리에 추가합니다.
                key = parts[0].strip()
                value = ":".join(parts[1:]).strip()  # 콜론(:)을 포함한 나머지 부분을 값으로 사용
                response_dict[key] = value

        print(response_dict)
        return response_dict

    except Exception as e:
        print(f"Error in extract_data: {e}")
        return {}

async def chat(context_vars: sk.ContextVariables) -> bool:
    try:
        user_input = input("User:> ")
        context_vars["user_input"] = user_input

        # Extract parameters from user input
        extracted_parameters = await extract_data(user_input, context_vars)

        # Print extracted parameters
        print(extracted_parameters)

        # OpenAI와 대화 시도
        answer = await kernel.run_async(chat_function, input_vars=context_vars)

    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input.lower() == "exit":
        print("\n\nExiting chat...")
        return False

    print(f"ChatBot:> {answer}")
    return True

async def main() -> None:
    context = sk.ContextVariables()
    context["chat_history"] = ""

    chatting = True
    while chatting:
        chatting = await chat(context)

if __name__ == "__main__":
    asyncio.run(main())