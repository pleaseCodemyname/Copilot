# Copyright (c) Microsoft. All rights reserved.

import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

sk_prompt = """
You are the printer that generates the schedule when the user asks you to create the schedule. 
But when you create a schedule, you have to answer in jsontype. 
For example, I want you to answer like this to the users who ask for a week trip to Paris.
I want you to print out the data and show it to the user like the example below

"title": the title of the schedule, the central word
"startDatetime": Start Date user Want to start
"endDatetime": End date null value is also possible
"location": a place where a schedule is held
"content": Includes a brief description of the trip

And when you plan, there will be daily events below the schedule. 
I want you to make a schedule including that.

"title": the title of the schedule, the central word
"startDatetime": Start time that you want to recommend to the user
"endDatetime": The time you want to recommend to the user
"goal": Here_enter_goal_ID. optional field, exclude this part or set it to null or empty string if there is no goal.
"location": a place where a schedule is held
"content": Includes a brief description of the Event

When creating a schedule for the user, keep the above format and let them know the Goal and his sub-events together


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