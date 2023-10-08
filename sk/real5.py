import semantic_kernel as sk
import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning import StepwisePlanner
from semantic_kernel.planning.stepwise_planner.stepwise_planner import (StepwisePlannerConfig,)
from semantic_kernel.connectors.search_engine import BingConnector

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)) 


BING_API_KEY = sk.bing_search_settings_from_dot_env()
connector = BingConnector(BING_API_KEY)
kernel.import_skill(WebSearchEngineSkill(connector), skill_name="WebSearch")

async def main():
    class WebSearchEngineSkill:
        """
        A search engine skill.
        """
        from semantic_kernel.orchestartion.sk_context import SKContext
        from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter

        def __init__(self, connector) -> None:
            self._connector = connector
        @sk_function(
            description="Performs a web search for a given query", name="searchAsync"
        )
        @sk_function_context_parameter(
            name="query",
            description="The search query",
        )
        async def search_async(self, query: str, context: SKContext) -> str:
            query = query or context.variables.get("query")[1]
            result = await self.connector.search_async(query, num_results=5, offset=0)
            return str(result)
    ask = """
    Where is the most popular place to hang out with a girl friend in Seoul?
    """
    result = await plan.invoke_async()
    print(result)
asyncio.runS(main())
    
