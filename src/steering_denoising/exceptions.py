"""Domain-specific errors with actionable messages."""


class SteeringDenoisingError(RuntimeError):
    """Base error for the project."""


class ConfigurationError(SteeringDenoisingError, ValueError):
    """Raised when an experiment configuration is inconsistent."""


class CompatibilityError(SteeringDenoisingError):
    """Raised when artifacts come from incompatible model/layer setups."""


class LeakageError(SteeringDenoisingError):
    """Raised when validation directions leak into structured training."""
