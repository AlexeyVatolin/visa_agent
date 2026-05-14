import os

from tqdm import tqdm as _tqdm
from tqdm.asyncio import tqdm_asyncio as _tqdm_asyncio

_TQDM_DISABLE = os.environ.get("CLAUDECODE") == "1"


def tqdm(*args, **kwargs):
    return _tqdm(*args, disable=_TQDM_DISABLE, **kwargs)


class async_tqdm(_tqdm_asyncio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable=_TQDM_DISABLE, **kwargs)
