import pytest
from graphrag.query.structured_search.global_search.search import GlobalSearch, GlobalSearchResult
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.llm.base import BaseLLM
from graphrag.query.context_builder.builders import GlobalContextBuilder

class MockLLM(BaseLLM):
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, streaming=False, **kwargs):
        self.call_count += 1
        return "mocked response"

    async def agenerate(self, messages, streaming=False, **kwargs):
        self.call_count += 1
        return "mocked response"

class MockContextBuilder(GlobalContextBuilder):
    def build_context(self, conversation_history=None, **kwargs):
        return ["mocked context"], {}

@pytest.fixture
def global_search():
    llm = MockLLM()
    context_builder = MockContextBuilder()
    return GlobalSearch(llm, context_builder)

def test_parse_search_response_valid(global_search):
    valid_response = '''
    {
        "points": [
            {"description": "Point 1", "score": 90},
            {"description": "Point 2", "score": 80}
        ]
    }
    '''
    result = global_search.parse_search_response(valid_response)
    assert len(result) == 2
    assert result[0] == {"answer": "Point 1", "score": 90}
    assert result[1] == {"answer": "Point 2", "score": 80}

def test_parse_search_response_invalid_json(global_search):
    invalid_json = "This is not JSON"
    with pytest.raises(ValueError, match="Failed to parse response as JSON"):
        global_search.parse_search_response(invalid_json)

def test_parse_search_response_missing_points(global_search):
    missing_points = '{"data": "No points here"}'
    with pytest.raises(ValueError, match="Response JSON does not contain a 'points' list"):
        global_search.parse_search_response(missing_points)

def test_parse_search_response_invalid_point_format(global_search):
    invalid_point = '''
    {
        "points": [
            {"wrong_key": "Point 1", "score": 90}
        ]
    }
    '''
    with pytest.raises(ValueError, match="Error processing points"):
        global_search.parse_search_response(invalid_point)

def test_parse_search_response_with_text_prefix(global_search):
    response_with_prefix = '''
    Here's the response:
    {
        "points": [
            {"description": "Point 1", "score": 90}
        ]
    }
    '''
    result = global_search.parse_search_response(response_with_prefix)
    assert len(result) == 1
    assert result[0] == {"answer": "Point 1", "score": 90}

def test_parse_search_response_non_integer_score(global_search):
    non_integer_score = '''
    {
        "points": [
            {"description": "Point 1", "score": "high"}
        ]
    }
    '''
    with pytest.raises(ValueError, match="Error processing points"):
        global_search.parse_search_response(non_integer_score)

@pytest.mark.asyncio
async def test_map_response_single_batch(global_search):
    context_data = "Test context"
    query = "Test query"
    result = await global_search._map_response_single_batch(context_data, query)
    assert isinstance(result, SearchResult)
    assert result.context_data == context_data
    assert result.context_text == context_data
    assert result.llm_calls == 1

@pytest.mark.asyncio
async def test_reduce_response(global_search):
    map_responses = [
        SearchResult(response=[{"answer": "Point 1", "score": 90}], context_data="", context_text="", completion_time=0, llm_calls=1, prompt_tokens=0),
        SearchResult(response=[{"answer": "Point 2", "score": 80}], context_data="", context_text="", completion_time=0, llm_calls=1, prompt_tokens=0),
    ]
    query = "Test query"
    result = await global_search._reduce_response(map_responses, query)
    assert isinstance(result, SearchResult)
    assert result.llm_calls == 1

@pytest.mark.asyncio
async def test_asearch(global_search):
    query = "Test query"
    result = await global_search.asearch(query)
    assert isinstance(result, GlobalSearchResult)
    assert result.llm_calls > 0
    assert global_search.llm.call_count > 0  # Access the mock LLM through the GlobalSearch instance
