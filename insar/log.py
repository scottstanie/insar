"""
This module exports a Log class that wraps the logging python package

Uses the standard python logging utilities, just provides
nice formatting out of the box.

Usage:

    from log import get_log
    logger = get_log()

    logger.info("Something happened")
    logger.warning("Something concerning happened")
    logger.error("Something bad happened")
    logger.critical("Something just awful happened")
    logger.debug("Extra printing we often don't need to see.")
    # Custom output for this module:
    logger.success("Something great happened: highlight this success")
"""
import logging

from colorlog import ColoredFormatter
import argparse


def get_log(debug=False, name=__file__):
    """Creates a nice log format for use across multiple files.

    Default logging level is INFO

    Args:
        name (Optional[str]): The name the logger will use when printing statements
        debug (Optional[bool]): If true, sets logging level to DEBUG

    """
    logger = logging.getLogger(name)
    return format_log(logger, debug=debug)


def format_log(logger, debug=False):
    """Makes the logging output pretty and colored with times"""
    log_level = logging.DEBUG if debug else logging.INFO

    if debug:
        format_ = '[%(asctime)s] [%(log_color)s%(levelname)s/%(process)d %(filename)s %(reset)s] %(message)s%(reset)s'
    else:
        format_ = '[%(asctime)s] [%(log_color)s%(levelname)s %(filename)s%(reset)s] %(message)s%(reset)s'
    formatter = ColoredFormatter(
        format_,
        datefmt='%m/%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'black,bg_red',
            'SUCCESS': 'white,bg_blue'
        })

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.SUCCESS = 25  # between WARNING and INFO
    logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(log_level)

        logger.info('Logger initialized: %s' % (logger.name, ))

    if debug:
        logger.setLevel(debug)

    return logger


if __name__ == '__main__':
    # Example usage
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true', required=False, help='Show debug output')

    args = p.parse_args()
    debug = args.debug or False

    log = get_log(debug=debug)

    log.critical('Sample critical')
    try:
        print(1 / 0)
    except ZeroDivisionError:
        log.exception('Sample exception')
        log.error('Sample error', exc_info=True)
    log.warning('Sample warning')
    log.success('Sample SUCCESS!')
    log.debug('Sample debug')
