# Copyright (c) Microsoft. All rights reserved.

import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know'
when it doesn't know the answer.
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

extract_prompt = """
To begin with,
you need to know how to distinguish parameters based on the values you enter. First, the default parameter is "title", "startDatime", "endDatime", "location", "content", "image" and "isCompleted".
title should include something describing its purpose, representative content, an action, or a person's name. 
startDatetime is An event, schedule, or schedule begins. Must specify "year-month-day-hour-minute" with no words. only datetime.
endDatime is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "year-month-day-hour-minutes" with no words. only datetime.
location means place or where to meet except time or date. 
I hope the content would be the users' input except time.
image value is always null.
isCompleted value begins with false unless users specify when they say or mention it is done or complete. They say or mention it is complete, change the value from false to true.
When answering, ignore special characters, symbols and blank spaces. Just answer title, startDatetime, endDatetime, location, content, image, and isCompleted.
You are gonna combine with the second step, so do not answer yet.

Second step is choose only one that fits well with the context of the conversation from users.

Here is 6 definition of words. 

QueryByTime: The user wants to find information about user's plan, and a time range is given.

QueryByQuery: The user wants to find information about user's plan, but no time range is provided.

CreatePlan: The user wants to create a plan. 

RecommandPlan: the user asks for recommendations for plan

UpdatePlan: The user wants to modify an user's plan.

DeletePlan: The user wants to delete an user's plan.

Specify which of words did you choose and add the word with the first answer.
The final output Data type always goes like this,
● 함수: the value will be the answer of six definitions.
● 데이터: From now on, values will be json-type of the second answer.
You must comply with this data-structure.

Third,
When "함수:" value is CreatePlan, print(계획에 추가하시겠습니까?) without explanation or additional words. Result would be only print value.
And then add the final output type goes like this,
"● the title of the second answer"
"● mm월 dd일 hh시 ~hh시"
"● 장소: location of the second answer"
or
When "함수:" value is UpdatePlan, print(계획을 수정하시겠습니까?) without explanation or additional words. Result would be only print value.
or
When "함수:" value is DeletePlan, print(계획을 삭제하시겠습니까?) without explanation or additional words. Result would be only print value.
{{$chat_history}}
User:> {{$user_input}}
"""

yn_prompt="""
In case of CreatePlan, if users' replies include "예, yes, 물론, 추가, of course, add", "title of the second answer" + 계획이 추가되었습니다.
"""

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", sk_oai.OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)

prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=2000, temperature=0.7, top_p=0.4
)

sk_template = sk.PromptTemplate(
    sk_prompt, kernel.prompt_template_engine, prompt_config
)

extract_template = sk.PromptTemplate(
    extract_prompt, kernel.prompt_template_engine, prompt_config
)

yn_template = sk.PromptTemplate(
    yn_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, sk_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

extract_function_config = sk.SemanticFunctionConfig(prompt_config, extract_template)
extract_function = kernel.register_semantic_function("EXTRACT", "Extract", extract_function_config)

yn_function_config = sk.SemanticFunctionConfig(prompt_config, yn_template)
yn_function = kernel.register_semantic_function("YN", "Yn", yn_function_config)

async def extract_data(user_input, context_vars):
    try:
        answer = await kernel.run_async(extract_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})
        return answer
    except Exception as e:
        print(f"Error in extract_data: {e}")
        return None

async def yes_no(user_input, context_vars):
    try:
        answer = await kernel.run_async(yn_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})
        return answer
    except Exception as e:
        print(f"Error in yes_no: {e}")
        return None

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
    
    # Run extract_function
    extracted_response = await extract_data(user_input, context_vars)
    
    if extracted_response:
        # Check if the extracted response is CreatePlan
        if extracted_response.get('함수') == 'CreatePlan':
            print("계획에 추가하시겠습니까?")
            # Wait for user input
            user_response = input("User:> ")
            
            # Check if user response includes positive confirmation
            if any(word in user_response.lower() for word in ["예", "yes", "물론", "추가", "of course", "add"]):
                print(f"{extracted_response.get('title')} 계획이 추가되었습니다.")
            else:
                print("추가가 취소되었습니다.")
                
        else:
            # Run chat_function for other cases
            answer = await kernel.run_async(chat_function, input_vars=context_vars)
            context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"
            print(f"ChatBot:> {answer}")
    else:
        # Run chat_function if no specific extraction is found
        answer = await kernel.run_async(chat_function, input_vars=context_vars)
        context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"
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
