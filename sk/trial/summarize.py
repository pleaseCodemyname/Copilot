import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextCompletion,
    OpenAITextCompletion,
)

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

prompt = """{{$input}}
Summarize the content above.
"""

summarize = kernel.create_semantic_function(
    prompt, max_tokens=2000, temperature=0.2, top_p=0.5
)

input_text = """
테슬라는 2003년 마틴 에버하드와 마크 타페닝이 창업한 회사로, 2004년 일론 머스크가 투자자로 참여했고, 초기 창업자들이 회사를 떠나면서 일론 머스크가 CEO가 되어 지금까지 회사를 이끌고 있다. 일론 머스크는 테슬라를 통해 전기차는 느리다는 편견을 깨려고 노력했고, 2018년 현재에는 스포츠카 모델인 로드스터, 세단 모델인 모델 S, SUV 모델인 모델 X, 준중형 모델인 모델 3까지 나온 상태이다. 기존의 전기차에 비해 배터리 용량도 크고, 충전 속도도 빨라 많은 사람들의 이목을 끌게 되었고, 반 자율주행 기술을 포함해서 많은 첨단 기술들이 적용된 덕분에 대표 모델인 모델 S는 프리미엄 전기차 세단이다.
"""

# If needed, async is available too: summary = await summarize.invoke_async(input_text)
summary = summarize(input_text)

print(summary)


# TLDR(Too long Don't Read, 내용 5단어로 줄여주는 기능)
sk_prompt = """

{{$input}}

Give me the TLDR in 5 words.
"""

text = """
    1) 정보처리기사: 국가 IT 기술 경쟁력 제고 및 급변하는 정보화 환경에 대처하기 위하여, 실무 중심의 업무 프로세스 기능 및 절차 측면의 해결 능력, 데이터베이스 설계 및 문제점 파악과 개선안 도출 등의 DB 실무 능력, 알고리즘 및 자료구조의 논리적 해결 능력, 급변하는 IT 환경에 대한 신기술 동향 파악 능력, 국제화에 대비한 전산 영어 실무 능력 등을 평가
    2) 네트워크관리사: 서버를 구축하고 보안 설정, 시스템 최적화 등 네트워크 구축 및 이를 효과적으로 관리할 수 있는 인터넷 관련 기술력에 대한 자격
    3) 리눅스마스터: 리눅스로 운영되는 전세계 80%이상의 스마트폰, 70%이상의 클라우드 서버, 세계 상위의 500대 슈퍼컴퓨터를 비롯해서 5세대 이동통신(5G), 사물인터넷(IoT), 드론, 자율주행차 등 미래성장동력 분야에서 다양한 응용기반기술에 토대가 되는 자격종목
"""

tldr_function = kernel.create_semantic_function(
    sk_prompt, max_tokens=200, temperature=0, top_p=0.5
)

summary = tldr_function(text)

print(f"Output: {summary}")
