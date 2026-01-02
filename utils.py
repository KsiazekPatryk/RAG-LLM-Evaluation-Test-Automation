import requests
import json
from pathlib import Path

def load_test_data(filename):
        project_directory = Path(__file__).parent.absolute()
        test_data_path = project_directory/"test_data"/filename
        with open(test_data_path) as f:
               return json.load(f)

#user_input= test_data["eval_sample"]["user_input"]

def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", 
                  json={
                        "question": test_data["question"],
                        "chat_history": [                           
                        ]
                    }).json()
    return responseDict


# Generate Rubrics based Criteria Scoring to evaluate LLM Response. This metric that is used to evaluate response. The rubric consists of descriptions for each score, typically
# ranging from 1 to 10. The response here is evalutaion based on score_descriptions and ground truth.

#rubrics = {
# "score1_description": The response is incorrect, irrelevant, or does not align with the ground truth.",
# "score2_description": The response partially matches the ground truth but includes significant errors,omissions, or irrelevant information."
# "score3_description": The response generally aligns with the ground truth but may lack detail, clarity or have mino inaccuracies."
# "score4_description": The response is mostly accurate and aligns well with the ground truth, with  only minor issues or missing details"
# "score5_description": The response is full accurate, aligns completely with the ground truth,and is clear and detailed.

#}

#user_input= "Where is the Eiffel Tower located?"
#response = "The Eiffel Tower is located in Europe and it is part of France"
#reference= "The Eiffel Tower is located in Paris"