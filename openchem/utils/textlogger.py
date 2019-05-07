import sys
import logging


def setup_textlogger(name, doprint=True, tofile=None):
    level = logging.DEBUG

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if doprint:
        # print to stream
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        if tofile is not None:
            # print to file
            file = logging.FileHandler(tofile)
            file.setLevel(level)
            file.setFormatter(formatter)
            logger.addHandler(file)

    return logger
