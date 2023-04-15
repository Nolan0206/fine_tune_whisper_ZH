import random
import os
from pathlib import Path
from typing import Generator

import pandas as pd
from colorama import Fore
from huggingface_hub import HfApi, HfFolder


def show_argparse(args):
    args_dict = vars(args)
    df = pd.DataFrame({'argument': args_dict.keys(), 'value': args_dict.values()})
    print('*' * 15)
    print(df)
    print('*' * 15)


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


def read_txt(txt_name: str) :
    try:
        txt = []
        print(txt_name)
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


'''
def read_dir_old(dir_name) -> list[str] | None:
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
        try:
            if os.path.splitext(file_name)[1] == '.json':
                json_file_example.append(file_name)
        except IndexError:
            pass
    else:
        if not json_file_example:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {dir_name}")
            print(Fore.BLUE + "Please check if the file directory is correct")
            sys.exit()
        else:
            return json_file_example
'''


def input_content() :
    content = input("the json folder")
    if not os.path.exists(content):
        print(Fore.RED + "Directory file is not exist")
        return None
    else:
        if os.path.isfile(content):
            print(Fore.RED + "file is not a dir")
            return None
        else:
            return content


def read_dir(dir_name) :
    if dir_name is None:
        print(Fore.RED + "Directory file is None")
        print(Fore.BLUE + "Please enter the json folder")
        json_directory = input_content()
    else:
        path_str = Path(dir_name)
        if path_str.exists() and path_str.is_dir():
            json_directory = dir_name
        else:
            json_directory = input_content()
    if isinstance(json_directory, str):
        path_str = Path(json_directory)
    else:
        return None
    json_file_example = []
    for n in path_str.glob('*.json'):
        json_file_example.append(str(n))
    else:
        if not json_file_example:
            print(Fore.RED + f"The json file list is empty, because the json file is not found in {dir_name}")
            print(Fore.BLUE + "Please check if the file directory is correct")
            return None
        else:
            return json_file_example


def read_dir_old2(dir_name) :
    if dir_name is None:
        print(Fore.RED + "Directory file is None")
        print(Fore.BLUE + "Please enter the json folder")
        json_directory = input_content()
    else:
        path_str = Path(dir_name)
        if path_str.exists() and path_str.is_dir():
            json_directory = dir_name
        else:
            json_directory = input_content()
    if isinstance(json_directory, str):
        path_str = Path(json_directory)
    else:
        return None
    json_file_example = path_str.glob('*.json')
    try:
        next(path_str.glob('*.json'))
        return json_file_example
    except StopIteration:
        print(Fore.RED + f"The json file list is empty, because the json file is not found in {dir_name}")
        print(Fore.BLUE + "Please check if the file directory is correct")
        return None


def login_hugging_face(token: str) :
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)
    print('We are logged in to Hugging Face now!')

    return None


def gen(any_list):
    for m in any_list:
        yield m
