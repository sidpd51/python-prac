import pytest

from study.loader import load

ex = load(__file__, "ex03_tool_registry")


@pytest.fixture(autouse=True)
def clean_registry():
    ex.TOOLS.clear()
    yield
    ex.TOOLS.clear()


def define_weather():
    @ex.tool
    def get_weather(city: str, unit: str = "celsius", days: int = 1) -> str:
        """
        Get the weather forecast for a city.

        Longer explanation the model doesn't need.
        """
        return f"{city}:{unit}:{days}"

    return get_weather


@pytest.mark.topic("py.decorators.syntax", "py.decorators.first_class")
def test_registers_and_returns_original_function():
    def search(query: str) -> list:
        return [query]

    decorated = ex.tool(search)
    assert decorated is search, "registration decorators return the function itself"
    assert {"search": search} == ex.TOOLS
    assert search("x") == ["x"]


@pytest.mark.topic("py.decorators.metadata")
def test_schema_from_signature_and_docstring():
    fn = define_weather()
    assert fn.schema == {
        "name": "get_weather",
        "description": "Get the weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    }


@pytest.mark.topic("py.decorators.metadata")
def test_schema_all_types_and_no_docstring():
    @ex.tool
    def f(a: str, b: int, c: float, d: bool, e: list, g: dict):
        return None

    props = f.schema["parameters"]["properties"]
    assert [p["type"] for p in props.values()] == [
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
    ]
    assert f.schema["description"] == ""
    assert f.schema["parameters"]["required"] == ["a", "b", "c", "d", "e", "g"]


@pytest.mark.topic("py.decorators.syntax")
def test_bad_tools_fail_at_decoration_time():
    def no_annotation(x):
        return x

    def varargs(*args: str):
        return args

    def kwargs(**kw: str):
        return kw

    def weird(x: set):
        return x

    for fn in (no_annotation, varargs, kwargs, weird):
        with pytest.raises(TypeError):
            ex.tool(fn)
    assert ex.TOOLS == {}


@pytest.mark.topic("py.decorators.syntax")
def test_duplicate_name_rejected():
    define_weather()
    with pytest.raises(ValueError):
        define_weather()


@pytest.mark.topic("py.decorators.higher_order")
def test_call_tool_binds_defaults_and_calls():
    define_weather()
    assert ex.call_tool("get_weather", {"city": "Pune"}) == "Pune:celsius:1"
    assert ex.call_tool("get_weather", {"city": "Oslo", "days": 3}) == "Oslo:celsius:3"


@pytest.mark.topic("py.decorators.higher_order")
def test_call_tool_rejects_bad_calls():
    define_weather()
    with pytest.raises(KeyError):
        ex.call_tool("delete_db", {})
    with pytest.raises(TypeError):
        ex.call_tool("get_weather", {})  # missing city
    with pytest.raises(TypeError):
        ex.call_tool("get_weather", {"city": "Pune", "country": "IN"})  # unexpected
    with pytest.raises(TypeError, match="days"):
        ex.call_tool("get_weather", {"city": "Pune", "days": "3"})
    with pytest.raises(TypeError, match="days"):
        ex.call_tool("get_weather", {"city": "Pune", "days": True})


@pytest.mark.topic("py.decorators.higher_order")
def test_call_tool_number_rules():
    @ex.tool
    def scale(x: float, factor: float = 2.0) -> float:
        return x * factor

    assert ex.call_tool("scale", {"x": 3}) == 6.0  # int ok for float
    with pytest.raises(TypeError):
        ex.call_tool("scale", {"x": False})
