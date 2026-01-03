import pytest
from ragas.metrics import RubricsScore
from ragas import SingleTurnSample


@pytest.mark.asyncio
async def test_rubric_score(llm_wrapper,getData):
    rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors,omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity or have mino inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with  only minor issues or missing details",
        "score5_description": "The response is full accurate, aligns completely with the ground truth,and is clear and detailed."

}

rubrics_score = RubricsScore(rubrics=rubrics, llm=llm_wrapper)
score = await rubrics_score.single_turn_ascore(getData)
print(score)
assert score > 6


@pytest.fixture
def getData():
    sample = SingleTurnSample(
        user_input= "Where is the Eiffel Tower located?",
        response = "The Eiffel Tower is located in Europe and it is part of France",
        reference= "The Eiffel Tower is located in Paris",
    )
    return sample



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