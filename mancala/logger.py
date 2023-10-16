from pathlib import Path

import logging
from logging import FileHandler, StreamHandler

from mancala.config import config


Path(config.LOG_DIRECTORY).mkdir(exist_ok=True)


FORMAT = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": %(message)s}'
ISO_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def Logger():
    logger = logging.getLogger("MANCALA LOG")
    logger.setLevel(config.LOG_LEVEL)

    formatter = logging.Formatter(fmt=FORMAT, datefmt=ISO_DATETIME_FORMAT)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_path = config.LOG_DIRECTORY + "debug.log"
    file_handler = FileHandler(file_path, encoding="utf-8")
    logger.addHandler(file_handler)

    return logger


def Playback(stream=False):
    logger = logging.getLogger("PLAYBACK")
    logger.setLevel(logging.DEBUG if config.PLAYBACK else logging.ERROR)

    formatter = logging.Formatter(fmt=FORMAT, datefmt=ISO_DATETIME_FORMAT)
    file_path = config.LOG_DIRECTORY + "playback.log"
    file_handler = FileHandler(file_path, encoding="utf-8")
    logger.addHandler(file_handler)

    if stream:
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = Logger()
playback = Playback()
