import argparse
from urllib.error import HTTPError
import pandas as pd
import os
import sys
import librosa
import json
import random
from huggingface_hub import HfApi, HfFolder
from colorama import Fore, init
import numpy as np
from datasets import load_dataset, interleave_datasets, concatenate_datasets, Audio, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer)

text_column_names = {'sentence', 'transcription', 'transciption'}  # possible text column choices
audio_column = 'audio'
text_column = 'sentence'
MAX_AUDIO_DURATION = 30  # because of whisper model input is 30 second, refer to paper
DEFAULT_SAMPLING_RATE = 16_000
MAX_LENGTH = 448

init(autoreset=True)


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def read_txt(txt_name: str) -> list[str] | None:
    try:
        txt = []
        with open(txt_name, "r") as f:
            line = f.readline()
            while line:
                txt.append(line.strip())
                line = f.readline()
    except FileNotFoundError:
        print("read_txt filename error")
        return None
    else:
        return txt


def read_dir(dir_name) -> list[str] | None:
    if dir_name is None:
        print(Fore.RED + "Directory file is None")
        print(Fore.BLUE + "Please enter the json folder")
        content = input("the json folder")
        if not os.path.exists(content):
            print(Fore.RED + "Directory file is not exist")
            sys.exit()
        else:
            if os.path.isfile(content):
                print(Fore.RED + "file is not a dir")
                sys.exit()
            else:
                json_directory = content
    else:
        json_directory = dir_name
    data_file_names = os.listdir(json_directory)
    json_file_example = []
    for file_name in data_file_names:
        if os.path.splitext(file_name)[1] == '.json':
            json_file_example.append(file_name)
    else:
        if not list:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {catalog_json}")
            print(Fore.BLUE + "Please check if the file directory is correct")
            sys.exit()
        else:
            return json_file_example


def read_json(json_name):
    with open(json_name, "r") as test:
        my_json = json.load(test)
        return my_json['audios']


def split_json(audio):
    seg_list, text_list = []
    for aid in range(len(audio)):
        segments = audio[aid]['segments']
        path_to_opus = audio[aid]['path']
        path_to_opus = path_to_opus.split('/')[-1]
        for seg_num in range(len(segments)):
            seg_begin = segments[seg_num]['begin_time']
            seg_end = segments[seg_num]['end_time']
            text = segments[seg_num]['text']
            voice, sr = librosa.load(path_to_opus, sr=16000, mono=True, offset=seg_begin,
                                     duration=seg_end - seg_begin)
            seg_list.append(voice)
            text_list.append(text)
    data_train, data_val = data_split(seg_list, ratio=float(args.train_ratio), shuffle=True)
    text_train, text_val = data_split(text_list, ratio=float(args.train_ratio), shuffle=True)
    '''
    dataset = {
        'audio': seg_list,
        'sentence': text_list
    }
    '''
    dataset_train = list(zip(data_train, text_train))
    dataset_val = list(zip(data_val, text_val))
    return dataset_train, dataset_val


# STEP  Loging to Hugging Face
# get your account token from https://huggingface.co/settings/tokens
def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)
    print('We are logged in to Hugging Face now!')

    return None


def push_dataset(path_to_hf, dataset, split):
    try:
        assert isinstance(split, str)
        dataset.push_to_hub(path_to_hf, split)
    except NameError or HTTPError:
        return False
    else:
        return True


def read_single_dataset(dataset_name, split, feature_extractor, tokenizer,
                        merge_audio_to_max):
    # gaijin
    common_voice = Dataset.load_from_disk(f'{dataset_name}/{split}')
    # make preprocessing here
    assert type(common_voice) == Dataset
    if merge_audio_to_max:
        print('[IMPORTANT] dataset size BEFORE merging:', common_voice.num_rows)
        mapper = get_batch_mapper_merging_max_duration(feature_extractor, tokenizer)
        common_voice = common_voice.map(mapper, batched=True, batch_size=128, remove_columns=list(common_voice.features)
                                        )
        print('[IMPORTANT] dataset size AFTER merging:', common_voice.num_rows)
    else:
        mapper = get_mapper(feature_extractor, tokenizer)
        common_voice = common_voice.map(mapper)
    return common_voice


def merge_datasets(dataset_dir, split, feature_extractor, tokenizer, merge_audio_to_max, interleave):
    ds_list = []
    for dataset_name in dataset_dir:
        # dataset_name, config, splits = dataset_name.split('|')
        # config = config if config else None
        ds = read_single_dataset(dataset_name, split, feature_extractor, tokenizer, merge_audio_to_max)
        ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else:  # just concat
        ds = concatenate_datasets(ds_list)

    return ds


def get_batch_mapper_merging_max_duration(
        feature_extractor: FeatureExtractionMixin,
        tokenizer: PreTrainedTokenizer):
    def mapper(batch):
        bs = len(batch[text_column])
        print(bs)
        result = {'input_features': [], 'labels': []}
        list_arr, list_text, total = [], [], 0
        for i in range(bs + 1):
            if i == bs or total + len(batch[audio_column][i]) / DEFAULT_SAMPLING_RATE > MAX_AUDIO_DURATION:
                if total == 0:
                    continue  # because it could be evenly distributed when i == bs
                tokens = tokenizer('.'.join(list_text)).input_ids
                if len(tokens) > MAX_LENGTH:
                    continue  # too long -> might mislead to not-aligning problem

                result['input_features'].append(
                    feature_extractor(list_arr, sampling_rate=DEFAULT_SAMPLING_RATE).input_features[0])
                result['labels'].append(tokens)
                list_arr, list_text, total = [], [], 0
            if i < bs:
                duration = len(batch[audio_column][i]) / DEFAULT_SAMPLING_RATE
                if duration > MAX_AUDIO_DURATION:
                    continue
                total += duration
                list_arr.append(batch[audio_column][i])
                list_text.append(batch[text_column][i])
        return result

    return mapper


def get_mapper(
        keep_chars: str,
        feature_extractor: FeatureExtractionMixin,
        tokenizer: PreTrainedTokenizer):
    def mapper(example):
        return {
            'input_features':
                feature_extractor(example[audio_column], sampling_rate=DEFAULT_SAMPLING_RATE).input_features[
                    0],
            'labels': tokenizer(text_column_normalize(example[text_column])).input_ids
        }

    return mapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_json', default=None, help='txt|catalog_json or dir|catalog_json')
    parser.add_argument('--train_ratio', default=0.8, help='Percentage of training set')
    parser.add_argument('--path_to_dataset_raw', default="./raw_dataset", help='Output path of the dataset')
    parser.add_argument('--path_to_dataset_revise', default="./revise_dataset",
                        help='Output path of the revised dataset')
    parser.add_argument('--push_to_hub', default=True, type=bool, help='push or not')
    parser.add_argument('--hf_token', default=None, type=str, help='User Login token')
    parser.add_argument('--repository_address', default='Nolan1206/Nolan_whisper_educate1',
                        help='Name of the repository')

    parser.add_argument('--path_to_whisper_hf', default=None, type=str, help='Local Configuration')
    parser.add_argument('--interleave', action='store_true', default=False, help='')
    parser.add_argument('--merge-audio-to-max', action='store_true', default=False,
                        help='if passed, then it will merge audios to `MAX_AUDIO_DURATION`')

    args = parser.parse_args()
    show_argparse(args)

    method, catalog_json = args.catalog_json.split('|')
    json_file_name = []
    if method == "txt":
        json_file_name = read_txt(catalog_json)
        if not json_file_name:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {catalog_json}")
            sys.exit()
    elif method == "dir":
        json_file_name = read_dir(catalog_json)
    else:
        print(Fore.RED + "error")
        sys.exit()

    for json_local_name in json_file_name:
        print(f"Processing {json_local_name}")
        audios = read_json(json_local_name)
        train_dataset, val_dataset = split_json(audios)
        train_dataset = pd.DataFrame(train_dataset, columns=['audio', 'sentence'])
        val_dataset = pd.DataFrame(val_dataset, columns=['audio', 'sentence'])
        train_dataset = Dataset.from_pandas(train_dataset)
        val_dataset = Dataset.from_pandas(val_dataset)
        train_dataset.save_to_disk(f"{args.path_to_dataset_raw}/train/{json_local_name.split('.')[0]}")
        print("owo===========train")
        val_dataset.save_to_disk(f"{args.path_to_dataset_raw}/val/{json_local_name.split('.')[0]}")
        print("owo===========val")
        if args.push_to_hub:
            login_hugging_face(args.hf_token)
            push_train_validation = push_dataset(path_to_hf=args.repository_address, dataset=train_dataset,
                                                 split='train')
            if push_train_validation:
                print("push Succeeded")
            else:
                print("push Failed")
            push_val_validation = push_dataset(path_to_hf=args.repository_address, dataset=train_dataset,
                                               split='validation')
            if push_val_validation:
                print("push Succeeded")
            else:
                print("push Failed")

    if args.path_to_whisper_hf is None:
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
    else:
        try:
            config = WhisperConfig.from_pretrained(args.path_to_whisper_hf)
            feature_extractor = WhisperFeatureExtractor.from_pretrained(args.path_to_whisper_hf)
            tokenizer = WhisperTokenizer.from_pretrained(args.path_to_whisper_hf)
            # processor = WhisperProcessor.from_pretrained(args.path_to_whisper_hf)
        except OSError or EnvironmentError:
            print(Fore.RED + "feature_extractor or tokenizer loads failure")
            sys.exit()
        else:
            print('owo==========feature_extractor and tokenizer')

    train_ds = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='train', feature_extractor=feature_extractor,
                              tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                              interleave=args.interleave)
    train_ds.save_to_disk(f"{args.path_to_dataset_revise}/train")
    print("owo===========train_revise")
    val_ds = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='val', feature_extractor=feature_extractor,
                            tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                            interleave=args.interleave)
    val_ds.save_to_disk(f"{args.path_to_dataset_revise}/val")
    print("owo===========val_revise")

'''
    train_ds = merge_datasets(
        args.train_datasets, args.interleave,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.use_cached_ds, args.merge_audio_to_max)

    eval_ds = merge_datasets(
        args.eval_datasets, False,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.use_cached_ds, args.merge_audio_to_max)
'''
# Preprocess


'''
    train_ds = merge_datasets(
        args.train_datasets, args.interleave,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.use_cached_ds, args.merge_audio_to_max)

    eval_ds = merge_datasets(
        args.eval_datasets, False,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.use_cached_ds, args.merge_audio_to_max)
'''
