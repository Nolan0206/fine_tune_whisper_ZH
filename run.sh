#! /bin/bash
echo "play"
python main.py --catalog_json txtwcatalog_json.txt --train_ratio 0.9 --path_to_dataset_raw /source/DataRepository/raw_dataset_simple --path_to_dataset_revise /source/DataRepository/revise_dataset/demo --json_record record_demo1.json --path_to_whisper_hf /source/DataRepository/whisper-hf_history --merge-audio-to-max True
