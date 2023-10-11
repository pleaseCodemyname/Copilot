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

get_prompt = """
You are a classifier that categorizes the input as either a Goal, an Event, or a Todo:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
Todo: Refers to a small task or duty that needs to be accomplished.
When answering, just answer either a Goal, Event or Todo. 
{{$chat_history}}
User:> {{$user_input}}
ChatBot:>
"""

crud_prompt = """
You are an action type recognizer that categorizes the input as either a create, read, update, or delete:
Create: Includes the act of meeting someone or doing something.
Read: Refers to the act of consuming information or data.
Update: Involves postpone, delay, modifying or changing existing information or data.
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


async def recognize_get_intent(user_input, context_vars):
    try:
        answer = await kernel.run_async(get_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})

        category = None
        if "goal" in answer.lower():
            category = "Goal"
        elif "event" in answer.lower():
            category = "Event"
        elif "todo" in answer.lower():
            category = "Todo"

        return category, answer.choices[0].text.strip()
    except Exception as e:
        return None, None

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

async def extract_data(user_input, context_vars):
    try:
        answer = await kernel.run_async(extract_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})

        extracted_data = {}
        split_answer_text = answer.choices[0].text.strip().split("\n")

        if "create" in context_vars['user_input'].lower():
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
            print("Processing 'read' data...")

        elif "update" in context_vars['user_input'].lower():
            print("Processing 'update' data...")

        elif "delete" in context_vars['user_input'].lower():
            print("Processing 'delete' data...")
    except Exception as e:
        print(f"Error extracting data: {e}")

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

    print(f"Get Intent: {get_intent}, Answer Text: {answer_text}")

    crud_intent = await recognize_crud_intent(user_input, context_vars)
    print(f"CRUD Intent: {crud_intent}")

    if get_intent:
        if crud_intent:
            print("Both Get and CRUD intents detected. Processing...")

            context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> Get Intent: {get_intent}, CRUD Intent: {crud_intent}\n"
            answer = await kernel.run_async(crud_function, input_vars=context_vars)
            context_vars["chat_history"] += f"ChatBot:> {answer}\n"

            await extract_data(user_input, context_vars)
        else:
            print("Only Get intent detected. Processing...")
            context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> Get Intent: {get_intent}\n"
            answer = await kernel.run_async(get_function, input_vars=context_vars)
            context_vars["chat_history"] += f"ChatBot:> {answer}\n"
    else:
        print("No Get intent detected. Performing general chat...")
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
