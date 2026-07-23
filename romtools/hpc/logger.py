class Logger:
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
        self.hostname = None
        self.debug_mode = False

    def debug(self, message, local=False):
        if self.debug_mode:
            self.log(f"[DEBUG] {message}", local=local)

    def log(self, message, local=False):
        if self.verbosity > 0:
            if self.hostname and not local:
                print(f"[{self.hostname}] {message}")
            else:
                print(message)

    def set_hostname(self, hostname):
        self.hostname = hostname
