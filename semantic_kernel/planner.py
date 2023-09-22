import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextCompletion,
    OpenAITextCompletion,
)
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.core_skills.text_skill import TextSkill


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

    planner = BasicPlanner()

    skills_directory = "skills/"
    summarize_skill = kernel.import_semantic_skill_from_directory(
        skills_directory, "SummarizeSkill"
    )
    writer_skill = kernel.import_semantic_skill_from_directory(
        skills_directory, "WriterSkill"
    )
    text_skill = kernel.import_skill(TextSkill(), "TextSkill")

    ask = """
    내일은 내 베프의 생일이야. 좋은 아이디어 없을까? 그 친구는 영국애라서 영어로 작성해줘.
    모든 문자를 대문자로 작성해줘"""

    original_plan = await planner.create_plan_async(ask, kernel)

    print(original_plan.generated_plan)

    sk_prompt = """
    {{$input}}
    Rewrite the above in the style of Shakespeare.
    """
    shakespeareFunction = kernel.create_semantic_function(
        sk_prompt, "shakespeare", "ShakespeareSkill", max_tokens=2000, temperature=0.8
    )
    ask = """
    내일은 내 베프의 생일이야. 그 친구는 영국인이고 제일 좋아하는 거는 전자기기야.
    영어로 작성해주고, 전자기기에 대한 선물 추천해줘.
    모든 문자를 대문자로 작성해줘"""
    new_plan = await planner.create_plan_async(ask, kernel)
    original_results = await planner.execute_plan_async(original_plan, kernel)
    print(original_results)
    new_results = await planner.execute_plan_async(new_plan, kernel)
    print(new_results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
