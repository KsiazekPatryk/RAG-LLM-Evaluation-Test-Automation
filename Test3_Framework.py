import os
from langchain_openai import ChatOpenAI
import pytest
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
import requests

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm=lang_chain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", 
                  json={
                        "question": question,
                        "chat_history": [                           
                        ]
                    }).json()
    
    sample = SingleTurnSample(
        user_input="How many articles are there in the Selenium webdriver python course?",
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference="23"
    )
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7