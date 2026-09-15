import logging

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """Write through tqdm so that the messages do not break the progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


# The console is handled here, the log file is handled by hydra.job_logging.
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
handler = TqdmHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
