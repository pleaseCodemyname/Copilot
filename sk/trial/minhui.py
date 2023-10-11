# Copyright (c) Microsoft. All rights reserved.

import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

sk_prompt = """
You are the printer that generates the schedule when the user asks you to create the schedule. 
You should recommend it and share the Event and Todo that are inherited from Goal.
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
To-Do: Refers to a small task or duty that needs to be accomplished.

There may or may not be a goal.
If a goal is created, a value is created using uuid, and there is only one goal. And this may or may not include Event or Todo.

And now

Event data format include eventType, title, startDatetime, endDatetime, goal, location and content. Make sure you make those results in a Json-Type.
Classify "eventType" : Event
"title" should contain something symbolic or representative of the value entered by the user.
"startDatetime" is An event, schedule, or schedule begins. Must specify "year-month-day-hour-minute" with no words. only datetime.
"endDatime" is when an event, schedule, or schedule ends. But when there is no endDatetime specified, add one hour from startDatetime. Must specify "year-month-day-hour-minutes" with no words. only datetime.
"goal" value is what you get from uuid. It can be null when user has no goal.
"location" means meeting someone or a place.
I hope the "content" includes anything other than behavioral and planning.

finally
Todo data format include eventType, title, goal, content, isCompleted. Make sure you make those results in a Json-Type.
Classify "eventType": Todo
"title" should contain something symbolic or representative of the value entered by the user.
"goal" value is what you get from uuid. It can be null when user has no goal.
I hope the "content" includes anything other than behavioral and planning.
"isCompleted" starts with false. But when users get the feeling or nuance that conveys the meaning that it is over, then change to true.

Depending on how many days the user has
Day1:
{ Contents
}
Day2:
{ Contents
}
Day3:
{ Contents
}
......
......
Can you answer me in this form?
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