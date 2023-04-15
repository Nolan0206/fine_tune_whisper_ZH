from typing import Any
import numpy
from transformers import PreTrainedTokenizer
from transformers.feature_extraction_utils import FeatureExtractionMixin


audio_column = 'audio'
text_column = 'sentence'
MAX_AUDIO_DURATION = 30  # because of whisper model input is 30 second, refer to paper
DEFAULT_SAMPLING_RATE = 16_000
MAX_LENGTH = 448


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
                    feature_extractor(numpy.concatenate(list_arr), sampling_rate=DEFAULT_SAMPLING_RATE).input_features[0])
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
        feature_extractor: FeatureExtractionMixin,
        tokenizer: PreTrainedTokenizer):
    def mapper(example):
        return {
            'input_features':
                feature_extractor(example[audio_column], sampling_rate=DEFAULT_SAMPLING_RATE).input_features[
                    0],
            'labels': tokenizer(example[text_column]).input_ids
        }

    return mapper
