import os
import sys
import librosa
import datasets
from datasets import load_dataset, DatasetDict, Dataset
#datasets_cache_dir = "/content/drive/MyDrive/Whisper/Datasets"
#os.environ["HF_DATASETS_CACHE"] = datasets_cache_dir
#os.chdir('/content/drive/MyDrive')

common_voice = DatasetDict()
common_voice= DatasetDict.load_from_disk("/mnt/common_voice_prepared")
print(common_voice)
print(common_voice['train'][0]['sentence'])

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
feature_extractor = WhisperFeatureExtractor.from_pretrained("/mnt/whisper-small-zh/")
tokenizer = WhisperTokenizer.from_pretrained("/mnt/whisper-small-zh/")
processor = WhisperProcessor.from_pretrained("/mnt/whisper-small-zh/")

'''
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                                  {input_str}")
print(f"Decoded  w/  special:        {decoded_with_special}")
print(f"Decoded  w/out  special:  {decoded_str}")
print(f"Are  equal:                          {input_str  ==  decoded_str}")
'''

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 模型评估
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("/mnt/whisper-small-zh/")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 定义训练配置
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=1, # Total number of training epochs to perform
    output_dir="/mnt/whisper-small-zh/result", # model predictions and checkpoints will be written on Google Drive to be able to recover checkpoints
    per_device_train_batch_size=32, # The batch size per GPU/TPU core/CPU for training
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5, # The initial learning rate for AdamW optimizer
    warmup_steps=100, # Number of steps used for a linear warmup from 0 to learning_rate
    gradient_checkpointing=True, # use gradient checkpointing to save memory at the expense of slower backward pass
    fp16=True, #  Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training
    evaluation_strategy="steps", # Evaluation is done (and logged) every eval_steps
    save_strategy="steps", # Save is done every save_steps
    per_device_eval_batch_size=16, # The batch size per GPU/TPU core/CPU for evaluation
    predict_with_generate=True, # Whether to use generate to calculate generative metrics
    generation_max_length=225, # The max_length to use on each evaluation loop
    save_steps=300, # Number of updates steps before two checkpoint saves
    save_total_limit=5, # will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
    eval_steps=300, # Number of update steps between two evaluations if evaluation_strategy="steps"
    logging_steps=25, # Number of update steps between two logs if logging_strategy="steps"
    report_to=["tensorboard"], # Report the results and logs to tensorboard
    load_best_model_at_end=True, # load the best model found during training at the end of training
    metric_for_best_model="wer", # specify the metric to use to compare two different models
    greater_is_better=False, # set it to False as our metric is better when lower
    push_to_hub=False, # push the model to the Hub every time the model is saved
)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,

)
trainer.train()


print("Saving fine-tuned model")
model.save_pretrained(save_directory='/mnt/whisper-small-zh/result/model21')
processor.save_pretrained(save_directory='/mnt/whisper-small-zh/result/model')