# from fastapi import FastAPI, HTTPException, Query
# from dotenv import load_dotenv
# import openai
# import os
# import uvicorn
# import requests

# load_dotenv()
# app = FastAPIs()

# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

#기본 템플릿
import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)) #Chat을 할 수 있게 하는 코드

#AzureOpenAI를 쓰거나 OpenAI를 쓸때 고르는 코드
# if useAzureOpenAI:
#     deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
#     kernel.add_chat_service("chat_completion", AzureChatCompletion(deployment, endpoint, api_key))
# else:
#     api_key, org_id = sk.openai_settings_from_dot_env()
#     kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

skils_directory = "myskills"

recFunctions = kernel.import_semantic_skill_from_directory(skils_directory, "skill_type")

travelFunctions = recFunctions["recommend"]

result = travelFunctions("오늘 서울에서 갈만한 곳 추천해주라")

print(result)

