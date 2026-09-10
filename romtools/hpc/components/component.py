"""Shared construction for the helpers that the dispatchers compose."""

from romtools.hpc.logger import Logger


class Component:
    """
    Base for the dispatcher's helpers: each carries the dispatcher's
    configuration and logger, and stands alone with defaults for both.

    Arguments:
        config: The dispatcher's configuration dictionary
        logger: An instance of the Logger class for logging
    """

    def __init__(self, *, config: dict = None, logger: Logger = None):
        self.config = config if config is not None else {}
        self.logger = logger if logger is not None else Logger()
