"""Safe, actionable errors exposed outside protocol adapters."""


class HammerCodeError(Exception):
    """Base class whose message is suitable for terminal display."""


class ConfigurationError(HammerCodeError):
    pass


class UntrustedEndpointError(ConfigurationError):
    pass


class ConversationError(HammerCodeError):
    pass


class ConversationBusyError(ConversationError):
    pass


class InvalidTurnStateError(ConversationError):
    pass


class ToolError(HammerCodeError):
    pass


class PermissionError(ToolError):
    pass


class RuleFileError(PermissionError):
    pass


class ModelClientError(HammerCodeError):
    pass


class AuthenticationError(ModelClientError):
    pass


class PermissionDeniedError(ModelClientError):
    pass


class RateLimitError(ModelClientError):
    pass


class InvalidRequestError(ModelClientError):
    pass


class ProviderUnavailableError(ModelClientError):
    pass


class TransportError(ModelClientError):
    pass


class RequestTimeoutError(ModelClientError):
    pass


class StreamProtocolError(ModelClientError):
    pass


class StreamInterruptedError(ModelClientError):
    pass
