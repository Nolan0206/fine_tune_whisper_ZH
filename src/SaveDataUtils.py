from urllib.error import HTTPError
from pathlib import Path
import pandas as pd
from datasets import Dataset


def push_dataset(path_to_hf, dataset, split, file_name: str):
    try:
        assert isinstance(split, str)
        dataset.push_to_hub(path_to_hf, split)
    except NameError or HTTPError or AssertionError:
        print(f"push {file_name.split('/')[-1]} ({split}) Failed")
    else:
        print(f"push {file_name.split('/')[-1]} ({split}) Succeeded")


def save_to_local_raw(dataset, raw_path, split, json_path_name):
    path_str = Path(json_path_name)
    json_name = path_str.stem
    dataset = pd.DataFrame(dataset, columns=['audio', 'sentence'])
    dataset = Dataset.from_pandas(dataset)
    dataset.save_to_disk(f"{raw_path}/{split}/{json_name}")
    print(f"owo==========={split}")
