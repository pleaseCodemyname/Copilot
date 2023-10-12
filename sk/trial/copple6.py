import asyncio
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

sk_prompt = """
ChatBot can have a conversation with you about any topic. Continue asking questions to the user until they specify a Goal, Event, or Todo. Once satisfied, guide them to provide information that satisfies one of the CRUD operations.
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
When answering, speak Naturally and answer either a goal, event or todo based on users' input.
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

crud_prompt = """
You are an action type recognizer that categorizes the input as either a create, read, update, or delete:
Through the user's conversation, understanding context-appropriate words, and then extracting active content in the next sentence, 
Create: Includes the act of meeting someone or doing something. And if the output is Create, ask "생성하시겠습니까?"
Read: Refers to the act of consuming information or data. And if the output is Read, ask "조회하시겠습니까?
Update: Involves postpone, delay, modifying or changing existing information or data. And if the output is Update, "수정 또는 변경하시겠습니까?"
Delete: Contains the meaning of deleting or making something disappear, Eradication, Elimination. And if the output is Delete, ask "삭제하시겠습니까?"
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

get_template = sk.PromptTemplate(
    get_prompt, kernel.prompt_template_engine, prompt_config
)

crud_template = sk.PromptTemplate(
    crud_prompt, kernel.prompt_template_engine, prompt_config
)

extract_template = sk.PromptTemplate(
    extract_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

get_function_config = sk.SemanticFunctionConfig(prompt_config, get_template)
get_function = kernel.register_semantic_function("GET", "Get", get_function_config)

crud_function_config = sk.SemanticFunctionConfig(prompt_config, crud_template)
crud_function = kernel.register_semantic_function("CRUD", "Crud", crud_function_config)

extract_function_config = sk.SemanticFunctionConfig(prompt_config, extract_template)
extract_function = kernel.register_semantic_function("EXTRACT", "Extract", extract_function_config)



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
    
    answer = await kernel.run_async(get_function, input_vars=context_vars)
    if not answer:
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
