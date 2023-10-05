import semantic_kernel as sk
import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.basic_planner import BasicPlanner

kernel = sk.Kernel()
planner = BasicPlanner()
api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)) 
async def main():
    ask = """   
    내일 여행갈껀데 어디로 갈지 추천해주라
    """

    skills_directory = "myskills/skill_type"
    summarize_skill= kernel.import_semantic_skill_from_directory(skills_directory, "SummarizeSkill")
    writer_skill = kernel.import_semantic_skill_from_directory(skills_directory, "WriterSkill")
    #내가 따로 추가하는 Native Functions
    text_skill = kernel.import_skill(TextSkill(), "TextSkill")

    basic_plan = await planner.create_plan_async(ask, kernel)

    print(basic_plan.generated_plan) # Planner가 내 질문을 받고, JSON-based plan으로 
asyncio.run(main())


