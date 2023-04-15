import librosa
import json
from colorama import Fore
from src.utils import data_split, read_txt, read_dir
from tqdm import tqdm
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool

from src.SaveDataUtils import push_dataset, save_to_local_raw
from src.utils import data_split, read_txt, read_dir, login_hugging_face

def read_json_config(catalog_json: str):
    try:
        method, catalog_json = catalog_json.split('w')
    except ValueError:
        print(f"{catalog_json} is an error format")
        return None
    if method == "txt":
        json_file_name = read_txt(catalog_json)
        if not json_file_name:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {catalog_json}")
            return None
        return json_file_name
    elif method == "dir":
        json_file_name = read_dir(catalog_json)
        if not json_file_name:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {catalog_json}")
            return None
        else:
            return json_file_name
    else:
        print(Fore.RED + "error")
        return None


def read_json(json_name):
    with open(json_name, "r") as test:
        my_json = json.load(test)
        return my_json['audios']


def split_json(audio, ratio_train):
    seg_list, text_list = [], []
    for aid in range(len(audio)):
        segments = audio[aid]['segments']
        path_to_opus = audio[aid]['path']
        # path_to_opus = path_to_opus.split('/')[-1]
        # path_to_opus = f'/source/DataRepository/{path_to_opus}'
        path_new = path_to_opus.split('/')[-2]
        path_ = Path(path_to_opus)
        path_new_ = path_.stem
        path_to_opus = f'/source/DataRepository/audio_wav/train/youtube/{path_new}/{path_new_}.wav'
        # print(path_to_opus)
        for seg_num in tqdm(range(len(segments)), desc = 'Current Json', colour = 'green'):
            seg_begin = segments[seg_num]['begin_time']
            seg_end = segments[seg_num]['end_time']
            text = segments[seg_num]['text']
            voice, sr = librosa.load(path_to_opus, sr=16000, mono=True, offset=seg_begin,
                                     duration=seg_end - seg_begin)
            seg_list.append(voice)
            text_list.append(text)
    data_train, data_val = data_split(seg_list, ratio=float(ratio_train), shuffle=True)
    text_train, text_val = data_split(text_list, ratio=float(ratio_train), shuffle=True)
    '''
    dataset = {
        'audio': seg_list,
        'sentence': text_list
    }
    '''
    dataset_train = list(zip(data_train, text_train))
    dataset_val = list(zip(data_val, text_val))
    return dataset_train, dataset_val
    
def multithreading_push_json(hf_token, repository_address, train_ratio, path_to_dataset_raw):
    def Push_hub(hf_token_, repository_address_, train_dataset_, val_dataset_, batch_):

        login_hugging_face(hf_token_)
        push_dataset(path_to_hf=repository_address_, dataset=train_dataset_, split='train',
                     file_name=batch_)
        push_dataset(path_to_hf=repository_address_, dataset=val_dataset_, split='validation',
                     file_name=batch_)


    def map_multithreading_push_json(batch):

        print(f"Processing {batch}")
        audios = read_json(batch)
        train_dataset, val_dataset = split_json(audios, train_ratio)
        save_to_local_raw(dataset=train_dataset, raw_path=path_to_dataset_raw, split="train",
                          json_path_name=batch)
        save_to_local_raw(dataset=val_dataset, raw_path=path_to_dataset_raw, split="validation",
                          json_path_name=batch)
        Push_hub(hf_token, repository_address, train_dataset, val_dataset, batch)
        print(f"Processing {batch} success")

    return map_multithreading_push_json


def multithreading_json(train_ratio, path_to_dataset_raw):
    def map_multithreading_json(batch):

        print(f"Processing {batch}")
        audios = read_json(batch)
        train_dataset, val_dataset = split_json(audios, train_ratio)
        save_to_local_raw(dataset=train_dataset, raw_path=path_to_dataset_raw, split="train",
                          json_path_name=batch)
        save_to_local_raw(dataset=val_dataset, raw_path=path_to_dataset_raw, split="validation",
                          json_path_name=batch)
        print(f"Processing {batch} success")

    return map_multithreading_json
