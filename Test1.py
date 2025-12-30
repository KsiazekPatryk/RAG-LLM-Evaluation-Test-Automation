from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
#user input --> query
#response -> response
#reference -> Ground truth
#retrived context -> Top k retreived documents


def test_context_precision():
    # create object of class for that specific metric

    #Power of LLM + method metric score
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    context_precision = LLMContextPrecisionWithoutReference()

    #Feed data - 

    
    #score 
