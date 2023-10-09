import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)) #Chat을 할 수 있게 하는 코드

prompt = """ {{$input}}
일정을 요약해줘
"""
summarize = kernel.create_semantic_function(prompt, max_tokens=500, temperature=0.9, top_p=0.5)

input_text= """
나는 오늘 친구랑 밖에서 술먹기로했다. 그런데 오늘 친구랑 약간 말다툼을 했다.
나는 세상을 살아가는데 있어서 돈이 최고라 했지만, 친구는 주변에 사람들이 많은게 더 중요하다고 했다.
물론 친구의 말에는 일리가 있지만 나는 돈이 없으면 할 수 있는게 줄어들고, 여유가 없어지는 걸 느껴봤기 때문에
뭔가 돈이 있는게 더 삶의 질이 좋아진다고 생각해서 그렇게 얘기한 것같다.
정답은 없지만... 돈 많이벌자!! 그리고 친구랑 다시 화해해야지...
"""

summary = summarize(input_text)

print(summary)