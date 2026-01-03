import os
import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
import requests
#user input --> query
#response -> response
#reference -> Ground truth
#retrived context -> Top k retreived documents

@pytest.mark.asyncio

async def test_context_precision():
    # create object of class for that specific metric
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    #Power of LLM + method metric score
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"

    #Feed data - 
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
    "question": "How many articles are there in the Selenium webdriver python course?",
    "chat_history": [
    ]
    }).json()
    
    print(responseDict)


    sample = SingleTurnSample(
        user_input=question,
        response=responseDict["answer"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],responseDict["retrieved_docs"][1]["page_content"],responseDict["retrieved_docs"][2]["page_content"]]

    )

    #Score
    score = await context_precision.single_turn_ascore(sample)
    print(score)
    assert score > 0.8

    #sample = SingleTurnSample(
    #    user_input="How many articiles are in the Selenium webdriver python course?",
    #    response= "There are 23 articles in the course.",
    #    retrieved_contexts=["Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design Stage\n This coruse includes:\17.5 hours on demand video\nAssignments\n23 articles\n9 donloadable resources\nAccess on mobile and TV\n Certificate of completion\nRequiremensts"]
    #)