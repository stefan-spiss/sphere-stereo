#
# Author: Stefan Spiss
#
import logging
import sys

__logger = logging.getLogger('omnimvs')
LOG_INFO = __logger.info
LOG_ERROR = __logger.error
LOG_WARNING = __logger.warning
LOG_DEBUG = __logger.debug
LOG_CRITICAL = __logger.critical


class ConsoleStreamHandler(logging.StreamHandler):
    def __init__(self, info_stream=sys.stdout, error_stream=sys.stderr):
        super().__init__(info_stream)
        self.info_stream = info_stream
        self.error_stream = error_stream

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.stream = self.error_stream
        else:
            self.stream = self.info_stream
        super().emit(record)

# Idea from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output?page=1&tab=scoredesc#tab-top
# help with chatgpt
class Formatter(logging.Formatter):
    def __init__(self, format_str='[%(levelname)s: %(asctime)s] %(message)s (%(filename)s:%(lineno)d)', datefmt='%y.%m.%d %H:%M:%S'):
        super().__init__(format_str, datefmt)
        grey = '\x1b[38;20m'
        yellow = '\x1b[33;20m'
        red = '\x1b[31;20m'
        bold_red = '\x1b[31;1m'
        reset = '\x1b[0m'
        self.FORMATS = {
            logging.DEBUG: grey + format_str + reset,
            logging.INFO: grey + format_str + reset,
            logging.WARNING: yellow + format_str + reset,
            logging.ERROR: red + format_str + reset,
            logging.CRITICAL: bold_red + format_str + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        self._style._fmt = log_fmt
        return super().format(record)


def initLogging(level):
    __logger.setLevel(level)

    console_stream_handler = ConsoleStreamHandler(sys.stdout, sys.stderr)
    console_stream_handler.setLevel(level)
    console_stream_handler.setFormatter(Formatter())

    # Add handlers to the logger
    __logger.addHandler(console_stream_handler)

def setupLogger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_stream_handler = ConsoleStreamHandler(sys.stdout, sys.stderr)
    console_stream_handler.setLevel(level)
    console_stream_handler.setFormatter(Formatter())

    # Add handlers to the logger
    logger.addHandler(console_stream_handler)
    return logger

