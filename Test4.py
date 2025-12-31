import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils import get_llm_response, load_test_data


@pytest.mark.parametrize("getData",load_test_data(),indirect=True)

@pytest.mark.asyncio
async def test_faitfulness(llm_wrapper,getData):
    faithful = Faithfulness(llm=llm_wrapper)
    score = faithful.single_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    
    sample = SingleTurnSample(
        user_input=test_data["question"]
        response=responseDict["answer"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"]
    )
    return sample