from slimx import tool
from slimx.tooling import execute_tool

@tool
def add(a: int, b: int) -> int:
    return a + b

def test_tool_exec():
    assert execute_tool(add, {"a": 2, "b": 3}) == 5
