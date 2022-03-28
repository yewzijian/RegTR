"""
Misc utilities
"""

import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys

import coloredlogs
import git


_logger = logging.getLogger(__name__)


def print_info(opt, log_dir=None):
    """ Logs source code configuration
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Print commit ID
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y-%m-%d')
        git_message = repo.head.object.message
        _logger.info('Source is from Commit {} ({}): {}'.format(git_sha[:8], git_date, git_message.strip()))

        # Also create diff file in the log directory
        if log_dir is not None:
            with open(os.path.join(log_dir, 'compareHead.diff'), 'w') as fid:
                subprocess.run(['git', 'diff'], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass

    # Arguments
    arg_str = ['{}: {}'.format(key, value) for key, value in vars(opt).items()]
    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


class DebugFileHandler(logging.FileHandler):
    """File handler that logs only debug messages"""
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == logging.DEBUG:
            return
        super().emit(record)


def prepare_logger(opt: argparse.Namespace, log_path: str = None):
    """Creates logging directory, and installs colorlogs

    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace

    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    """

    if log_path is None:
        if opt.dev:
            log_path = '../logdev'
            shutil.rmtree(log_path, ignore_errors=True)
        else:
            datetime_str = datetime.now().strftime('%y%m%d_%H%M%S')
            if opt.name is not None:
                log_path = os.path.join(opt.logdir, datetime_str + '_' + opt.name)
            else:
                log_path = os.path.join(opt.logdir, datetime_str)
    os.makedirs(log_path, exist_ok=True)

    fmt = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    datefmt = '%m/%d %H:%M:%S'

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Log to output stream
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(coloredlogs.ColoredFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream_handler)

    # Log to file also
    log_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler = logging.FileHandler(f'{log_path}/log.txt')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Log debug messages into another file to avoid cluttering the standard log file
    log_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler = DebugFileHandler(f'{log_path}/debug_logs.txt')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info('Output and logs will be saved to {}'.format(log_path))
    print_info(opt, log_path)

    return logger, log_path


def pretty_time_delta(seconds):
    """Pretty print a time delta in Python in days, hours, minutes and seconds

    Taken from: https://gist.github.com/thatalextaylor/7408395
    """
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)