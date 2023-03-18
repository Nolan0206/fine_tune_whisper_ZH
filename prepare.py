import os
import sys
import librosa
import datasets
from datasets import load_dataset, DatasetDict, Dataset, Audio
#datasets_cache_dir = "/content/drive/MyDrive/Whisper/Datasets"
#os.environ["HF_DATASETS_CACHE"] = datasets_cache_dir
#os.chdir('/content/drive/MyDrive')

common_voice = DatasetDict()
common_voice["train"]= Dataset.load_from_disk("/mnt/common_voice/train")
common_voice["test"]= Dataset.load_from_disk("/mnt/common_voice/test")
print(common_voice)

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
feature_extractor = WhisperFeatureExtractor.from_pretrained("/mnt/whisper-small-zh/")
tokenizer = WhisperTokenizer.from_pretrained("/mnt/whisper-small-zh/")
#processor = WhisperProcessor.from_pretrained("/mnt/whisper-small-zh/")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
print('=================')
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    #batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=3)
common_voice.save_to_disk('/mnt/common_voice_prepared')