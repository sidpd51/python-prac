import copy
import json
from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from study.loader import load

ex = load(__file__, "ex06_tool_schema")


class Address(BaseModel):
    city: str = Field(description="City name")
    country: str = "US"


class GetWeather(BaseModel):
    """Get the current weather for a location.

    Use for questions about temperature or rain.
    """

    location: Address = Field(description="Where to look")
    unit: Literal["c", "f"] = "c"
    title: str | None = None  # a PROPERTY called "title"
    extras: dict[str, int] = {}


DOC = "Get the current weather for a location.\n\nUse for questions about temperature or rain."


@pytest.mark.topic("py.pydantic.json_schema", "py.pydantic.api_validation")
def test_tool_definition_exact():
    assert ex.tool_definition(GetWeather) == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": DOC,
            "parameters": {
                "description": DOC,
                "type": "object",
                "additionalProperties": False,
                "required": ["location"],
                "properties": {
                    "location": {
                        "description": "Where to look",
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["city"],
                        "properties": {
                            "city": {"description": "City name", "type": "string"},
                            "country": {"default": "US", "type": "string"},
                        },
                    },
                    "unit": {"default": "c", "enum": ["c", "f"], "type": "string"},
                    "title": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                    "extras": {
                        "additionalProperties": {"type": "integer"},
                        "default": {},
                        "type": "object",
                    },
                },
            },
        },
    }


@pytest.mark.topic("py.pydantic.json_schema")
def test_no_refs_or_titles_anywhere():
    text = json.dumps(ex.tool_definition(GetWeather)["function"]["parameters"])
    assert "$ref" not in text and "$defs" not in text
    assert '"title": {' in text  # the property survives
    assert '"title": "' not in text  # ...but no title keywords


@pytest.mark.topic("py.pydantic.json_schema")
def test_names_and_overrides():
    class SearchKBArticles(BaseModel):
        query: str

    d = ex.tool_definition(SearchKBArticles)["function"]
    assert d["name"] == "search_kb_articles"
    assert d["description"] == ""
    d = ex.tool_definition(SearchKBArticles, name="kb", description="Search the KB")["function"]
    assert (d["name"], d["description"]) == ("kb", "Search the KB")


@pytest.mark.topic("py.pydantic.json_schema", "py.pydantic.nested")
def test_shared_def_used_twice_and_lists():
    class Route(BaseModel):
        origin: Address
        stops: list[Address] = []

    params = ex.tool_definition(Route)["function"]["parameters"]
    assert params["properties"]["origin"]["properties"]["city"] == {
        "description": "City name",
        "type": "string",
    }
    assert params["properties"]["stops"]["items"]["required"] == ["city"]
    # inlined copies must be independent objects
    params["properties"]["origin"]["properties"].clear()
    assert params["properties"]["stops"]["items"]["properties"]


@pytest.mark.topic("py.pydantic.json_schema")
def test_data_values_untouched():
    class Filter(BaseModel):
        meta: dict[str, str] = Field(default={"title": "keep me"}, examples=[{"title": "x"}])

    prop = ex.tool_definition(Filter)["function"]["parameters"]["properties"]["meta"]
    assert prop["default"] == {"title": "keep me"}
    assert prop["examples"] == [{"title": "x"}]


@pytest.mark.topic("py.pydantic.json_schema")
def test_recursive_model_rejected():
    class Node(BaseModel):
        name: str
        children: list["Node"] = []

    with pytest.raises(ValueError):
        ex.tool_definition(Node)


@pytest.mark.topic("py.pydantic.json_schema")
def test_definition_is_stable_across_calls():
    first = copy.deepcopy(ex.tool_definition(GetWeather))
    assert ex.tool_definition(GetWeather) == first


def search_docs(
    query: Annotated[str, Field(description="What to search for", min_length=1)],
    top_k: Annotated[int, Field(ge=1, le=20)] = 5,
    source: Optional[str] = None,  # noqa: UP045 - old-style spelling on purpose
) -> list[str]:
    """Search the internal knowledge base."""
    return [f"{query}:{top_k}:{source}"]


@pytest.mark.topic("py.pydantic.models", "py.pydantic.field")
def test_tool_from_function_builds_model():
    tool = ex.tool_from_function(search_docs)
    assert isinstance(tool, ex.Tool)
    assert (tool.name, tool.description) == ("search_docs", "Search the internal knowledge base.")
    schema = tool.args_model.model_json_schema()
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["description"] == "What to search for"
    assert schema["properties"]["top_k"]["maximum"] == 20
    assert schema["properties"]["top_k"]["default"] == 5
    assert schema["additionalProperties"] is False


@pytest.mark.topic("py.pydantic.models")
def test_tool_from_function_rejects_undescribable_signatures():
    def untyped(x, y: int):
        return x

    def star(*args: int):
        return args

    def kw(**kwargs: int):
        return kwargs

    for fn in (untyped, star, kw):
        with pytest.raises(TypeError):
            ex.tool_from_function(fn)


@pytest.mark.topic("py.pydantic.model_validate", "py.pydantic.api_validation")
def test_invoke_tool_validates_llm_arguments():
    tool = ex.tool_from_function(search_docs)
    assert ex.invoke_tool(tool, '{"query": "refund policy"}') == ["refund policy:5:None"]
    assert ex.invoke_tool(tool, {"query": "q", "top_k": "3", "source": "wiki"}) == ["q:3:wiki"]
    for bad in ('{"query": ""}', '{"query": "q", "top_k": 50}', '{"query": "q", "k": 1}', "{"):
        with pytest.raises(ValidationError):
            ex.invoke_tool(tool, bad)


@pytest.mark.topic("py.pydantic.nested", "py.pydantic.model_validate")
def test_invoke_tool_passes_model_instances():
    seen = []

    def book_trip(destination: Address, nights: int = 1) -> str:
        seen.append(destination)
        return f"{destination.city}/{nights}"

    tool = ex.tool_from_function(book_trip)
    assert ex.invoke_tool(tool, '{"destination": {"city": "Pune"}, "nights": 2}') == "Pune/2"
    assert isinstance(seen[0], Address) and seen[0].country == "US"
    defn = ex.tool_definition(tool.args_model, name=tool.name, description=tool.description)
    assert defn["function"]["parameters"]["properties"]["destination"]["required"] == ["city"]
