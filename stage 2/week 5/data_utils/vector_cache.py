import os
from pathlib import Path


def get_vector_cache_path(model_path: str, cache_dir: str = 'cache') -> str:
    base_name = Path(model_path).stem
    os.makedirs(cache_dir, exist_ok=True)
    return str(Path(cache_dir) / f'vectors_{base_name}.npy')
