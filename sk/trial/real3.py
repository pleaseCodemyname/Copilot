import semantic_kernel as sk
import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)) 

async def main():
    sk_prompt = """
    Chatbot who speaks politely and warm. Be kind to Users.
    When you are being asked, try to answer accurate information to users as much as you can.
    If you don't know the answer, just say "Sorry, I don't have the information"
    {{$history}}
    User: "{{$user_input}}"
    ChatBot: 
    """
    chat_function = kernel.create_semantic_function(sk_prompt, "ChatBot", max_tokens=500, temperature=0.9, top_p=0.5)
    context = kernel.create_new_context()
    context["history"] = ""
    context["user_input"] = "Hi, I'm thinking of taking a journey for a week. Please recommend any place to go"
    bot_answer = await chat_function.invoke_async(context=context)
    print(bot_answer)

    context["history"] += f"\nUser: {context['user_input']}\nChatBot: {bot_answer}\n"
    print(context["history"])

    async def chat (input_text: str) -> None:
        # Save new message in the context variables
        print(f"User: {input_text}")
        context["user_input"] = input_text
        
        # Process the user message and get an answer
        answer = await chat_function.invoke_async(context=context) 

        # Show the response
        print(f"ChatBot: {answer}")

        # Apped the new interaction to the chat history
        context["history"] += f'\nUser: {input_text}\nChatBot: {answer}\n'
    await chat("강릉은 어때?")
    await chat("2박3일 숙박할 호텔이나 펜션도 알아봐줄 수 있어?")
    await chat("비용은 한 30만원 이내가 좋을 것 같아.")
    print(context["history"])
asyncio.run(main())

# context["history"] += f"\nUser: {context['user_input']}\nChatBot: {bot_answer}\n"
# print(context["history"])
