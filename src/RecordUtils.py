import json
import os


def open_json_record(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as load_f:
            return json.load(load_f)
    except FileNotFoundError:
        print("read_txt filename error")
        return {}


def write_json_record(file_path: str, record_dict: dict) -> None:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as load_f:
            dict_bak = json.load(load_f)
        first_name, last_name = file_path.split(".")
        file_bak_path = f"{first_name}_bak.{last_name}"
        with open(file_bak_path, "w", encoding='utf-8') as f:
            json.dump(dict_bak, f, indent=4, sort_keys=True, ensure_ascii=False)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(record_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
        print("write_json_record success")
    else:
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(record_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
        print("write_json_record new success")


def update_list(json_name, history_list: list):
    for i in json_name:
        if i not in history_list:
            history_list.append(i)
    return history_list


def record_data(history_dict: dict, json_name: list, push_bool, map_bool) -> dict:
    if not history_dict:
        raw, revise, push_to_hub = [[] for _ in range(3)]
    else:
        raw = history_dict.get('raw', [])
        revise = history_dict.get('revise', [])
        push_to_hub = history_dict.get('push_to_hub', [])
    if push_bool and map_bool:
        raw = update_list(json_name, raw)
        push_to_hub = update_list(json_name, push_to_hub)
        revise = update_list(json_name, revise)
    elif push_bool:
        raw = update_list(json_name, raw)
        push_to_hub = update_list(json_name, push_to_hub)
    elif map_bool:
        raw = update_list(json_name, raw)
        revise = update_list(json_name, revise)
    else:
        raw = update_list(json_name, raw)
    return {
        "raw": raw,
        "revise": revise,
        "push_to_hub": push_to_hub,
    }


def find_new_json(history_dict: dict, json_name: list, category) -> list:
    if not history_dict:
        category_list = []
    else:
        category_list = history_dict.get(category, [])
    new_json_list = []
    for i in json_name:
        if i not in category_list:
            new_json_list.append(i)
    return new_json_list
