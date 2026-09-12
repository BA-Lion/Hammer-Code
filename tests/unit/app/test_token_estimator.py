from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.domain.events import ToolDefinition
from hammer_code.domain.messages import Message, ProviderStateBlock, Role, TextBlock, ToolCallBlock


def test_estimator_is_deterministic_for_ascii_cjk_json_and_tools() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_text("") == 0
    assert estimator.estimate_text("abc") == 1
    assert estimator.estimate_text("中文") > 0
    message = Message(
        Role.USER,
        (
            TextBlock("中文"),
            ToolCallBlock("call", "tool", {"x": 1}, '{"x":1}'),
            ProviderStateBlock("test", "opaque", {"b": 1, "a": "x"}),
        ),
    )
    tool = ToolDefinition("tool", "description", {"b": 1, "a": "x"})
    assert estimator.estimate_messages((message,)) > estimator.estimate_text("中文")
    assert estimator.estimate_tools((tool,)) == estimator.estimate_tools((tool,))
    assert estimator.estimate_request("system", (message,), (tool,)) > 0
