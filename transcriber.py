#!/usr/bin/env python3

import sys
import whisper

result_file = "outputs/results.txt"

model = whisper.load_model(sys.argv[1])

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("outputs/audio.mp3")
audio_trim = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio_trim, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language and write to results file
_, probs = model.detect_language(mel)
language_detected = max(probs, key=probs.get)
print(f"Detected language: {language_detected}")

with open(result_file, "a") as file:
    file.write(f"\nDetected language: {language_detected}\n")

# decode the audio
result = model.transcribe(audio)

# write the recognized text to file
with open(result_file, "a") as file:
    file.write("\nTranscription:\n")
    file.write(result["text"])
    file.write("\n")

print(f"Transcription saved at: {result_file}")
