import requests
import json

def load_test_data(filename):
        test_data_path = "/Users/patrykksiazek/RAG-LLM-Evaluation-Test-Automation/RAG-LLM-Evaluation-Test-Automation/testdata/Test3_Framework.json"
        with open(test_data_path) as f:
               return json.load(f)

def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", 
                  json={
                        "question": test_data["question"],
                        "chat_history": [                           
                        ]
                    }).json()
    return responseDict