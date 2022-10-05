#!/usr/bin/env python3

import youtube_dl
import subprocess
import whisper
import torch
import sys

audiofile = "audio.mp3"

# Download mp3 audio of a Youtube video. Credit to Stokry
# https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
def audio():
    video_info = youtube_dl.YoutubeDL().extract_info(url = sys.argv[1],download=False)
    options={
        'format':'bestaudio/best',
        'keepvideo':False,
        'outtmpl':audiofile,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

# Check wether CUDA is available or not
def checkDevice():
    if (torch.cuda.is_available() == 1):
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    return DEVICE

def main():
    audio() # Download an mp3 audio file to transcribe to text

	# Select speech recognition model
    name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    model = whisper.load_model(name,device=checkDevice())

    # Save transcribed text to file
    result = model.transcribe(audiofile)
    with open('transcription.txt', 'a') as file:
        file.write(result["text"])
        file.write("\n")

if __name__ == "__main__":
    main()
