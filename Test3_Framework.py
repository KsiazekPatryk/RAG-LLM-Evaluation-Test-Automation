import os
from langchain_openai import ChatOpenAI
import pytest
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
import requests

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

@pytest.mark.asyncio
@pytest.mark.parametrize("getData", 
                         [
                             {
                                 "question" : "How many articles are there in the Selenium webdriver python course?",
                                 "reference" : "23"
                             }
                         ],indirect=True
)
async def test_context_recall(llm_wrapper,getData):
    question = "How many articles are there in the Selenium webdriver python course?"

    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(getData)
    print(score)
    assert score > 0.7

   
@pytest.fixture
def getData(request):
    test_data = request.param

    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", 
                  json={
                        "question": test_data["question"],
                        "chat_history": [                           
                        ]
                    }).json()
    
    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference=test_data["reference"],
    )
    return sample