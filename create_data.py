import argparse
import sys
from itertools import chain

from src import *
import warnings

warnings.filterwarnings('ignore')
JSON_RECORD = './history/record.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_json', required=True, default=None, help='txt|catalog_json or dir|catalog_json')
    parser.add_argument('--train_ratio', default=0.8, help='Percentage of training set')
    parser.add_argument('--path_to_dataset_raw', default="./raw_dataset", help='Output path of the dataset')
    parser.add_argument('--push_to_hub', default=False, type=bool, help='push or not')
    parser.add_argument('--hf_token', default=None, type=str, help='User Login token')
    parser.add_argument('--repository_address', default='Nolan1206/Nolan_whisper_educate1',
                        help='Name of the repository')

    args = parser.parse_args()
    show_argparse(args)

    json_file_name = read_json_config(args.catalog_json)
    if json_file_name is None:
        sys.exit()

    history_json_dict = open_json_record(JSON_RECORD)

    # 记录已处理的json文件，方便添加数据
    # raw_name_written = gen(any_list=json_file_name)
    new_json_raw_file = find_new_json(history_dict=history_json_dict, json_name=json_file_name, category="raw")

    if args.push_to_hub:
        new_json_push_file = find_new_json(history_dict=history_json_dict, json_name=json_file_name,
                                           category="push_to_hub")
        full_set = set(chain(new_json_raw_file, new_json_push_file))
    else:
        full_set = set(new_json_raw_file)

    for json_local_name in tqdm(full_set, desc='Parsing Json', colour='red'):
        print(f"Processing {json_local_name}")
        audios = read_json(json_local_name)
        train_dataset, val_dataset = split_json(audios, args.train_ratio)
        save_to_local_raw(dataset=train_dataset, raw_path=args.path_to_dataset_raw, split="train",
                          json_path_name=json_local_name)
        save_to_local_raw(dataset=val_dataset, raw_path=args.path_to_dataset_raw, split="validation",
                          json_path_name=json_local_name)
        if args.push_to_hub:
            login_hugging_face(args.hf_token)
            push_dataset(path_to_hf=args.repository_address, dataset=train_dataset, split='train',
                         file_name=json_local_name)
            push_dataset(path_to_hf=args.repository_address, dataset=train_dataset, split='validation',
                         file_name=json_local_name)

    new_json_dict = record_data(history_dict=history_json_dict, json_name=json_file_name, push_bool=args.push_to_hub)
    write_json_record(JSON_RECORD, new_json_dict)
    print("owo===========create_data finished")
