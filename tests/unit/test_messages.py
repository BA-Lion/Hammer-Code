import pytest

from hammer_code.domain.messages import Message, ProviderStateBlock, Role, TextBlock


def test_message_requires_ordered_non_empty_content() -> None:
    message = Message(Role.USER, (TextBlock("hello"),))
    assert isinstance(message.content[0], TextBlock)
    assert message.content[0].text == "hello"
    with pytest.raises(ValueError):
        Message(Role.USER, ())


def test_opaque_provider_state_never_reprs_its_data() -> None:
    block = ProviderStateBlock("anthropic_messages", "signature", {"signature": "secret"})
    assert "secret" not in repr(block)
