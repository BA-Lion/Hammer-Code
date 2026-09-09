from hammer_code.errors import TransportError
from hammer_code.llm._common import map_exception


class APIConnectionError(Exception):
    pass


class ConnectError(Exception):
    pass


class ProxyError(Exception):
    pass


def test_connection_errors_are_actionable_and_do_not_leak_raw_details() -> None:
    secret = "token=secret https://example.test/path?key=secret"
    for exception_type in (APIConnectionError, ConnectionError, ConnectError, ProxyError):
        mapped = map_exception(exception_type(secret))
        assert isinstance(mapped, TransportError)
        assert str(mapped) == "Model connection failed; check network, proxy, DNS and TLS settings"
        assert secret not in str(mapped)
