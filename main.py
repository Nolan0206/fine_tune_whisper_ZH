import argparse
import sys
from itertools import chain
from colorama import init
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor)

# 从本地导入
'''
from src.MapData import *
from src.LoadDataUtils import *
from src.ParsingJson import *
from src.RecordUtils import *
from src.SaveDataUtils import *
from src.utils import *
'''
from src import *
import warnings

init(autoreset=True)

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_json', default=None, help='txtwcatalog_json or dirwcatalog_json')
    parser.add_argument('--train_ratio', default=0.8, help='Percentage of training set')
    parser.add_argument('--path_to_dataset_raw', default="./raw_dataset", help='Output path of the dataset')
    parser.add_argument('--path_to_dataset_revise', default="./revise_dataset",
                        help='Output path of the revised dataset')
    parser.add_argument('--push_to_hub', default=False, type=bool, help='push or not')
    parser.add_argument('--hf_token', default=None, type=str, help='User Login token')
    parser.add_argument('--repository_address', default='Nolan1206/Nolan_whisper_educate1',
                        help='Name of the repository')
    parser.add_argument('--json_record', default=None, type=str, help='User Login token')
    parser.add_argument('--map_data', default=False,  type=bool, help='map or not')
    parser.add_argument('--path_to_whisper_hf', default=None, type=str, help='Local Configuration')
    parser.add_argument('--path_to_record', default=None, type=str, help='The json file that has been written')
    parser.add_argument('--interleave',  default=False, help='')
    parser.add_argument('--multithreading', default=False,
                        help='Whether multithreading is enabled')
    parser.add_argument('--merge-audio-to-max',  default=False,
                        help='if passed, then it will merge audios to `MAX_AUDIO_DURATION`')

    args = parser.parse_args()
    show_argparse(args)

    json_file_name = read_json_config(args.catalog_json)
    print(json_file_name)
    if json_file_name is None:
        sys.exit()

    history_json_dict = open_json_record(args.json_record)

    # 记录已处理的json文件，方便添加数据
    # raw_name_written = gen(any_list=json_file_name)
    new_json_raw_file = find_new_json(history_dict=history_json_dict, json_name=json_file_name, category="raw")

    if args.push_to_hub:
        new_json_push_file = find_new_json(history_dict=history_json_dict, json_name=json_file_name,
                                           category="push_to_hub")
        full_set = set(chain(new_json_raw_file, new_json_push_file))
    else:
        full_set = set(new_json_raw_file)
    if args.multithreading:
        p = Pool(processes=min(len(full_set), os.cpu_count()))
        if args.push_to_hub:
            parse_json = multithreading_push_json(hf_token=args.hf_token, repository_address=args.repository_address,
                                                  train_ratio=args.train_ratio,
                                                  path_to_dataset_raw=args.path_to_dataset_raw)
            p.map(parse_json, full_set)
            p.close()
            p.join()
        else:
            parse_json = multithreading_json(train_ratio=args.train_ratio,path_to_dataset_raw=args.path_to_dataset_raw)
            p.map(parse_json, full_set)
            p.close()
            p.join()
    else:
        for json_local_name in full_set:
            print(f"Processing {json_local_name}")
            audios = read_json(json_local_name)
            train_dataset, val_dataset = split_json(audios, args.train_ratio)
            save_to_local_raw(dataset=train_dataset, raw_path=args.path_to_dataset_raw, split="train",
                              json_path_name=json_local_name)
            save_to_local_raw(dataset=val_dataset, raw_path=args.path_to_dataset_raw, split="validation",
                              json_path_name=json_local_name)
            # =================
            if args.push_to_hub:
                login_hugging_face(args.hf_token)
                push_dataset(path_to_hf=args.repository_address, dataset=train_dataset, split='train',
                             file_name=json_local_name)
                push_dataset(path_to_hf=args.repository_address, dataset=val_dataset, split='validation',
                             file_name=json_local_name)


    if args.map_data:
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

        train_ds = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='train',
                                  feature_extractor=feature_extractor,
                                  tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                                  interleave=args.interleave, history_json=history_json_dict.get("revise", []))
        train_ds.save_to_disk(f"{args.path_to_dataset_revise}/train")
        print("owo===========train_revise")
        val_ds = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='validation',
                                feature_extractor=feature_extractor,
                                tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                                interleave=args.interleave, history_json=history_json_dict.get("revise", []))
        val_ds.save_to_disk(f"{args.path_to_dataset_revise}/val")
        print("owo===========val_revise")

    new_json_dict = record_data(history_json_dict, json_file_name, args.push_to_hub, args.map_data)
    write_json_record(args.json_record, new_json_dict)
    print("owo===================finished")
