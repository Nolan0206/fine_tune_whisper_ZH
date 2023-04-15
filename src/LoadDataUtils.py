from pathlib import Path

from colorama import Fore
from datasets import interleave_datasets, concatenate_datasets, Dataset
import os
from src.MapData import get_mapper, get_batch_mapper_merging_max_duration


def read_single_dataset(dataset_name, feature_extractor, tokenizer,
                        merge_audio_to_max):
    common_voice = Dataset.load_from_disk(dataset_name)
    # make preprocessing here
    assert type(common_voice) == Dataset
    if merge_audio_to_max:
        print('[IMPORTANT] dataset size BEFORE merging:', common_voice.num_rows)
        mapper = get_batch_mapper_merging_max_duration(feature_extractor, tokenizer)
        common_voice = common_voice.map(mapper, batched=True, batch_size=128,
                                        remove_columns=list(common_voice.features)
                                        )
        print('[IMPORTANT] dataset size AFTER merging:', common_voice.num_rows)
    else:
        mapper = get_mapper(feature_extractor, tokenizer)
        common_voice = common_voice.map(mapper)
    return common_voice


def merge_datasets_old(dataset_dir, split, feature_extractor, tokenizer, merge_audio_to_max, interleave):
    ds_list = []
    dataset_list = os.listdir(f'{dataset_dir}/{split}')
    for dataset_name in dataset_list:
        if os.path.isdir(dataset_name):
            # dataset_name, config, splits = dataset_name.split('|')
            # config = config if config else None
            ds = read_single_dataset(dataset_name, feature_extractor, tokenizer, merge_audio_to_max)
            ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else:  # just concat
        ds = concatenate_datasets(ds_list)

    return ds


def merge_datasets(dataset_dir, split, feature_extractor, tokenizer, merge_audio_to_max, interleave, history_json):
    ds_path_list, ds_list = [], []
    path_str = Path(f'{dataset_dir}/{split}')
    for n in path_str.glob('*'):
        if (n.is_dir()) and (str(n) not in history_json):
            ds_path_list.append(str(n))
    else:
        if not ds_path_list:
            print(Fore.RED + f"The raw file list is empty, because the raw file is not found in {dataset_dir}/{split}")
            print(Fore.BLUE + "Please check if the file directory is correct")
            return None
        elif len(ds_path_list) == 1:
            return read_single_dataset(ds_path_list[0], feature_extractor, tokenizer, merge_audio_to_max)

    for dataset_name in ds_path_list:
        # dataset_name, config, splits = dataset_name.split('|')
        # config = config if config else None
        ds = read_single_dataset(dataset_name, feature_extractor, tokenizer, merge_audio_to_max)
        ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else:  # just concat
        ds = concatenate_datasets(ds_list)

    return ds


def merge_datasets_test(dataset_dir, split, feature_extractor, tokenizer, merge_audio_to_max, interleave):
    ds_list = []
    path_str = Path(f'{dataset_dir}/{split}')
    path_generator = path_str.glob('*')
    try:
        next(path_generator)
    except StopIteration:
        print(Fore.RED + f"The raw file list is empty, because the raw file is not found in {dataset_dir}/{split}")
        print(Fore.BLUE + "Please check if the file directory is correct")
        return None
    try:
        next(path_generator)
    except StopIteration:
        return read_single_dataset(next(path_str.glob('*')), feature_extractor, tokenizer, merge_audio_to_max)
    else:
        for dataset_name in path_str.glob('*'):
            ds = read_single_dataset(dataset_name, feature_extractor, tokenizer, merge_audio_to_max)
            ds_list.append(ds)

        if interleave:
            ds = interleave_datasets(ds_list, seed=42)
        else:  # just concat
            ds = concatenate_datasets(ds_list)

        return ds


def load_merge_datasets(dataset_dir_list, split: str, interleave: bool, preprocessed: bool,feature_extractor, tokenizer, merge_audio_to_max):
    if dataset_dir_list:
        ds_list = []
        for dataset_dir in dataset_dir_list:
            dataset_dir = dataset_dir.rstrip('/')
            path_to_data = f"{dataset_dir}/{split}"
            if not os.path.isdir(path_to_data):
                print(f"Sorry, I cannot find the <{path_to_data}>, Check that the path and type are correct")
                return None
            else:
                if preprocessed:
                    ds = Dataset.load_from_disk(path_to_data)
                    ds_list.append(ds)
                else:
                    ds = read_single_dataset(path_to_data, feature_extractor, tokenizer, merge_audio_to_max)
                    ds_list.append(ds)
    else:
        print(Fore.RED + "The list of incoming files is empty")
        return None

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else:  # just concat
        ds = concatenate_datasets(ds_list)
    return ds
