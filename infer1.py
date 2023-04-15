import torch
import whisper
from transformers import pipeline

#MODEL_PATH = "/mnt/whisper-small-zh/"
MODEL_PATH = "/source/DataRepository/whisper-hf_history"
AUDIO_PATH = "12.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_PATH,
    chunk_length_s=30,
    device=device,
    ignore_warning=True
)

# we override any special forced tokens for auto language detection - not necessary if you use transformers from main!
all_special_ids = pipe.tokenizer.all_special_ids
transcribe_token_id = all_special_ids[-5]
pipe.model.config.forced_decoder_ids = [[2, transcribe_token_id]]

# inference
out = pipe(AUDIO_PATH)["text"]
print(out)
