import semantic_kernel as sk
from semantic_kernel.core_skills import TimeSkill
import asyncio


async def get_today():
    kernel = sk.Kernel()

    time = kernel.import_skill(TimeSkill())
    result = await kernel.run_async(time["today"])

    print(result)


if __name__ == "__main__":
    asyncio.run(get_today())
