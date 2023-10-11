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

crud_prompt = """
Here is 4 definition of words. 

CreatePlan: The user wants to create a plan or do something, hangout with someone, play with others, planning of doing something active. 

RecommandPlan: the user asks for recommendations for plan

UpdatePlan: The user wants to modify an user's plan.

DeletePlan: The user wants to delete an user's plan.

Choose only one that fits well with the context of the conversation from users and it's gonna be the first answer. 

Next,
Find the value corresponding to the first answer below 4 Plans
When the first answer is CreatePlan, just ask "$user_input" + "계획을 추가하시겠어요?." 
or
When UpdatePlan, jsut ask "$user_input" + "계획을 수정하시겠습니까?"
or
When DeletePlan, just ask "$user_input" + "계획을 삭제하시겠습니까?"
or 
When RecommendPlan, just ask "$user_input" +"계획을 추천받으시겠습니까?" 
{{$chat_history}}
User:> {{$user_input}}
"""

yn_prompt="""
When users say, express or deliver positive meaning such as "yes, 응, of course, 
If the previous conversation was:
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

crud_template = sk.PromptTemplate(
    crud_prompt, kernel.prompt_template_engine, prompt_config
)

yn_template = sk.PromptTemplate(
    yn_prompt, kernel.prompt_template_engine, prompt_config
)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

crud_function_config = sk.SemanticFunctionConfig(prompt_config, crud_template)
crud_function = kernel.register_semantic_function("Crud", "CRUD", crud_function_config)

yn_function_config = sk.SemanticFunctionConfig(prompt_config, yn_template)
yn_function = kernel.register_semantic_function("YN", "Yn", yn_function_config)

async def extract_data(user_input, context_vars):
    try:
        answer = await kernel.run_async(extract_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})
    except Exception as e:
        print(f"Error in extract_data: {e}")

async def yes_no(user_input, context_vars):
    try:
        answer = await kernel.run_async(yn_function, input_vars={'$chat_history': context_vars['chat_history'], '$user_input': user_input})
    except Exception as e:
        print(f"Error in yes_no: {e}")

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
    
    answer = await kernel.run_async(crud_function, input_vars=context_vars)
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
