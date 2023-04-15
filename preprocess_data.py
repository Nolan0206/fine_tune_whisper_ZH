import argparse
import sys

from colorama import init
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor)

from src import *
import warnings

warnings.filterwarnings('ignore')

init(autoreset=True)

JSON_RECORD = './history/record.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset_raw', default="./raw_dataset", help='Output path of the dataset')
    parser.add_argument('--path_to_dataset_revise', default="./revise_dataset",
                        help='Output path of the revised dataset')
    parser.add_argument('--path_to_whisper_hf', default=None, type=str, help='Local Configuration')
    parser.add_argument('--interleave', action='store_true', default=False, help='')
    parser.add_argument('--merge-audio-to-max', action='store_true', default=False,
                        help='if passed, then it will merge audios to `MAX_AUDIO_DURATION`')

    args = parser.parse_args()
    show_argparse(args)

    history_json_dict = open_json_record(JSON_RECORD)

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

    train_ds, ds_path_list = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='train',
                                            feature_extractor=feature_extractor,
                                            tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                                            interleave=args.interleave,
                                            history_json=history_json_dict.get("revise", []))
    train_ds.save_to_disk(f"{args.path_to_dataset_revise}/train")
    print("owo===========train_revise")
    val_ds, _ = merge_datasets(dataset_dir=args.path_to_dataset_raw, split='validation',
                               feature_extractor=feature_extractor,
                               tokenizer=tokenizer, merge_audio_to_max=args.merge_audio_to_max,
                               interleave=args.interleave, history_json=history_json_dict.get("revise", []))
    val_ds.save_to_disk(f"{args.path_to_dataset_revise}/validation")
    print("owo===========val_revise")

    new_json_dict = record_data(history_dict=history_json_dict, ds_path_list=ds_path_list, map_bool=True)
    write_json_record(args.json_record, new_json_dict)
    print("owo===================preprocess data finished")
