# import semantic_kernel as sk
# from semantic_kernel.connectors.ai.open_ai import (
#     AzureTextCompletion,
#     OpenAITextCompletion,
# )

# kernel = sk.Kernel()

# useAzureOpenAI = False

# # Configure AI service used by the kernel
# if useAzureOpenAI:
#     deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
#     kernel.add_text_completion_service(
#         "dv", AzureTextCompletion(deployment, endpoint, api_key)
#     )
# else:
#     api_key, org_id = sk.openai_settings_from_dot_env()
#     kernel.add_text_completion_service(
#         "dv", OpenAITextCompletion("text-davinci-003", api_key, org_id)
#     )

# sk_prompt = """
# ChatBot can have a conversation with you about any topic.
# It can give explicit instructions or say 'I don't know' if it does not have an answer.

# {{$history}}
# User: {{$user_input}}
# ChatBot: """

# chat_function = kernel.create_semantic_function(
#     sk_prompt, "ChatBot", max_tokens=2000, temperature=0.7, top_p=0.5
# )

# context = kernel.create_new_context()
# context["history"] = ""

# context["user_input"] = "서울 데이트 추천해줘!!"
# bot_answer = await chat_function.invoke_async(context=context)
# print(bot_answer)

# context["history"] += f"\nUser: {context['user_input']}\nChatBot: {bot_answer}\n"
# print(context["history"])


# async def chat(input_text: str) -> None:
#     # Save new message in the context variables
#     print(f"User: {input_text}")
#     context["user_input"] = input_text

#     # Process the user message and get an answer
#     answer = await chat_function.invoke_async(context=context)

#     # Show the response
#     print(f"ChatBot: {answer}")

#     # Append the new interaction to the chat history
#     context["history"] += f"\nUser: {input_text}\nChatBot: {answer}\n"

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextCompletion,
    OpenAITextCompletion,
)


async def main():
    kernel = sk.Kernel()

    useAzureOpenAI = False

    # Configure AI service used by the kernel
    if useAzureOpenAI:
        deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service(
            "dv", AzureTextCompletion(deployment, endpoint, api_key)
        )
    else:
        api_key, org_id = sk.openai_settings_from_dot_env()
        kernel.add_text_completion_service(
            "dv", OpenAITextCompletion("text-davinci-003", api_key, org_id)
        )

    sk_prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if it does not have an answer.

    {{$history}}
    User: {{$user_input}}
    ChatBot: """

    chat_function = kernel.create_semantic_function(
        sk_prompt, "ChatBot", max_tokens=2000, temperature=0.7, top_p=0.5
    )

    context = kernel.create_new_context()
    context["history"] = ""

    async def chat(input_text: str) -> None:
        nonlocal context
        # Save new message in the context variables
        print(f"User: {input_text}")
        context["user_input"] = input_text

        # Process the user message and get an answer
        answer = await chat_function.invoke_async(context=context)

        # Show the response
        print(f"ChatBot: {answer}")

        # Append the new interaction to the chat history
        context["history"] += f"\nUser: {input_text}\nChatBot: {answer}\n"

    # Example conversation
    await chat("서울 데이트 추천해줘!!")
    await chat("가까운 식당 추천해줘.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
