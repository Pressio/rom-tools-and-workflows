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


class CampaignComponent(Component):
    """
    Base for the helpers that act on a campaign's directories.

    One rule decides what "no run directory" means, so submitting a job and
    staging its inputs cannot disagree about where the work happens.

    Arguments:
        campaign_directory: The directory work happens in when given no run_directory
    """

    def __init__(self, *, campaign_directory: str = None, config: dict = None,
                 logger: Logger = None):
        super().__init__(config=config, logger=logger)
        self.campaign_directory = campaign_directory

    def job_directory(self, run_directory: str = None) -> str:
        """Where work happens: the directory given, otherwise this campaign's."""
        return run_directory or self.campaign_directory
