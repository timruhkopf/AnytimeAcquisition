# register resolvers for omegaconf
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

import logging

logger = logging.getLogger(__name__)

def get_git_hash(_):
    try:
        # Retrieve the short git hash
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()
        return git_hash
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to retrieve git hash: {e}")
        return "unknown"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "error"


def resolve_path(relative_path: str):
    REPO_PATH = str(Path(__file__).parents[2])
    if relative_path == '':
        return REPO_PATH
    return REPO_PATH + '/' + relative_path


def parse_target_cls(string):
    return string.split('.')[-1]


# Register the resolver
OmegaConf.register_new_resolver("git_commit", get_git_hash)
OmegaConf.register_new_resolver("resolve_path", resolve_path)
OmegaConf.register_new_resolver("parse_target_cls", parse_target_cls)
OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
OmegaConf.register_new_resolver("int_mult", lambda x, y: int(x * y))
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("len", lambda x: len(x))
