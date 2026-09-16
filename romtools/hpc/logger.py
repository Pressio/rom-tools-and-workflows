class Logger:
    """
    Prints dispatcher output, tagging remote messages with the hostname.

    Messages logged through debug() appear only when debug logging is enabled.
    """

    def __init__(self, debug: bool = False):
        self.hostname = None
        self.debug_enabled = debug

    def debug(self, message, local=False):
        if self.debug_enabled:
            self.log(f"[DEBUG] {message}", local=local)

    def log(self, message, local=False):
        if self.hostname and not local:
            print(f"[{self.hostname}] {message}")
        else:
            print(message)

    def set_hostname(self, hostname):
        self.hostname = hostname
