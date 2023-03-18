# fine_tune_whisper_ZH
For local training and saving of small models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--eval_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--interleave', action='store_true', help='')
    parser.add_argument('--whisper-size', default='small')
    parser.add_argument('--language', default='mn,Mongolian', help='acronym,Full Language Name')
    parser.add_argument('--keep-chars', default=KEEP_CHARS, help='characters that would stay during preprocessing')
    parser.add_argument('--train-batch-size', default=32, type=int)
    parser.add_argument('--eval-batch-size', default=16, type=int)
    parser.add_argument('--max-steps', default=1000, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--version', default=1, type=int)

    # for reading and writing preprocessed dataset
    parser.add_argument('--hf-username', type=str, required=True)
    parser.add_argument('--use-cached-ds', action='store_true',
                        help='if passed, it will try to read from preprocessed dataset handle')
    parser.add_argument('--merge-audio-to-max', action='store_true',
                        help='if passed, then it will merge audios to `dataset_utils.MAX_AUDIO_DURATION`')

    # Trainer.train()
    parser.add_argument('--resume-from-checkpoint', action='store_true',
                        help='if passed, training will start from the latest checkpoint')

    args = parser.parse_args()
    show_argparse(args)