import whisper
model = whisper.load_model('./whisper-small-checkpoint.pt')
result = model.transcribe('123.wav') # probably longer than 10 min? hour?
print(result['text'])
