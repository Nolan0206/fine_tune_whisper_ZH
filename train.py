import argparse
import sys
import torch
import gc
from datasets import load_dataset, DatasetDict, Dataset
from colorama import init
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperProcessor)

# 从本地导入
from src import *

init(autoreset=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog_json', default='txtwtrain.txt', help='txtwcatalog_json or dirwcatalog_json')
    parser.add_argument('--path_to_whisper_hf', default='/source/DataRepository/whisper-hf_history', type=str, help='Local Configuration')
    parser.add_argument('--model_path', default='/source/DataRepository/whisper_hf_old', type=str, help='Local Configuration')
    parser.add_argument('--output_dir', default='./Training_results_demo', type=str, help='Local Configuration')
    parser.add_argument('--epochs', default=1, type=int, help='Local Configuration')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Local Configuration')
    parser.add_argument('--eval_batch_size', default=16, type=int, help='Local Configuration')
    parser.add_argument('--num_workers', default=8, type=int, help='Local Configuration')
    parser.add_argument('--resume_from_checkpoint', default=False, help='')
    parser.add_argument('--interleave', default=False, help='')
    parser.add_argument('--preprocessed', default=True, help='')
    parser.add_argument('--merge-audio-to-max', default=True,
                        help='if passed, then it will merge audios to `MAX_AUDIO_DURATION`')

    args = parser.parse_args()
    show_argparse(args)

    if args.path_to_whisper_hf is None:
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
    else:
        try:
            config = WhisperConfig.from_pretrained(args.path_to_whisper_hf)
            feature_extractor = WhisperFeatureExtractor.from_pretrained(args.path_to_whisper_hf)
            tokenizer = WhisperTokenizer.from_pretrained(args.path_to_whisper_hf)
            processor = WhisperProcessor.from_pretrained(args.path_to_whisper_hf)
        except OSError or EnvironmentError:
            print(Fore.RED + "feature_extractor or tokenizer loads failure")
            sys.exit()
        else:
            print('owo==========feature_extractor and tokenizer')

    dir_list = read_json_config(args.catalog_json)
    if dir_list is None:
        sys.exit()
    '''
    train_ds = load_merge_datasets(dataset_dir_list=dir_list, split='train', feature_extractor=feature_extractor,
                                   preprocessed=args.preprocessed, merge_audio_to_max=args.merge_audio_to_max,
                                   tokenizer=tokenizer, interleave=args.interleave)

    val_ds = load_merge_datasets(dataset_dir_list=dir_list, split='val', feature_extractor=feature_extractor,
                                 preprocessed=args.preprocessed, merge_audio_to_max=args.merge_audio_to_max,
                                 tokenizer=tokenizer, interleave=args.interleave)
    '''
    common_voice = DatasetDict()
    common_voice= DatasetDict.load_from_disk("/source/DataRepository/revise_dataset/simple")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # not compatible with gradient checkpointing

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = get_compute_metrics_func(tokenizer)

    # 定义训练配置

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.epochs,  # Total number of training epochs to perform
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,  # The batch size per GPU/TPU core/CPU for training
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,  # The initial learning rate for AdamW optimizer
        warmup_steps=100,  # Number of steps used for a linear warmup from 0 to learning_rate
        gradient_checkpointing=True,  # use gradient checkpointing to save memory at the expense of slower backward pass
        fp16=True,  # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training
        evaluation_strategy="steps",  # Evaluation is done (and logged) every eval_steps
        save_strategy="steps",  # Save is done every save_steps
        per_device_eval_batch_size=args.eval_batch_size,  # The batch size per GPU/TPU core/CPU for evaluation
        predict_with_generate=True,  # Whether to use generate to calculate generative metrics
        generation_max_length=225,  # The max_length to use on each evaluation loop
        save_steps=12920,  # Number of updates steps before two checkpoint saves
        save_total_limit=5,  # will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
        eval_steps=12920,  # Number of update steps between two evaluations if evaluation_strategy="steps"
        logging_steps=500,  # Number of update steps between two logs if logging_strategy="steps"
        report_to=["tensorboard"],  # Report the results and logs to tensorboard
        load_best_model_at_end=True,  # load the best model found during training at the end of training
        metric_for_best_model="wer",  # specify the metric to use to compare two different models
        greater_is_better=False,  # set it to False as our metric is better when lower
        push_to_hub=False,  # push the model to the Hub every time the model is saved
        remove_unused_columns=False,  # important when we use set_transform
        dataloader_num_workers=args.num_workers
    )
    # dataloader_num_workers=args.num_workers
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    try:
        #gc.collect()
        #torch.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPTED! Starting evaluation with current state')
        trainer.is_in_train = False

    metrics = evaluate_and_save(trainer, tokenizer, feature_extractor)

    # print("Saving fine-tuned model")
    # model.save_pretrained(save_directory='/mnt/whisper-small-zh/result/model/')
    # processor.save_pretrained(save_directory='/mnt/whisper-small-zh/result/processor/')
