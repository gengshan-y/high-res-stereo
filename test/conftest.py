import os

import pytest

from utils import sync_open_dataset


@pytest.fixture(scope='session')
def _sync_data():
    dataset_path = os.path.join(os.path.dirname(__file__), '../data')
    print(dataset_path)
    sync_open_dataset(dataset_path, tiny=True)
