#!/usr/bin/env python3

# Usage: python3 transcriber.py -u, --url <URL>

from googletrans import Translator
import youtube_dl
import subprocess
import whisper
import getopt
import torch
import sys
import re

audiofile = "audio.mp3" # Save audio file as audio.mp3

# Download mp3 audio of a Youtube video. Credit to Stokry
# https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
def getAudio():
    url = None
    argv = sys.argv[1:]
    try:
        opts,args = getopt.getopt(argv, "u:",["url="])
    except :
        print("Usage: python3 transcriber.py -u <url>")

    for opt, arg in opts:
        if opt in ['-u', '--url']:
            url = arg

    video_info = youtube_dl.YoutubeDL().extract_info(url=url,download=False)
    options={
        'format':'bestaudio/best',
        'keepvideo':False,
        'outtmpl':audiofile,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

# Check CUDA availability
def checkDevice():
    if (torch.cuda.is_available() == 1):
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    return DEVICE

def translateResult(orgFile, transFile, text):
    # Translate transcribed text. Credit to Harsh Jain at educative.io
    # https://www.educative.io/answers/how-do-you-translate-text-using-python
    translator = Translator() # Create an instance of Translator() class
    with open(orgFile, 'r') as transcription:
        contents = transcription.read()
        translation = translator.translate(contents)
    with open(transFile, 'a') as file:
        file.write(translation.text)

# Put a newline character after each sentence
def formatResult(fileName, text):
    formatText = re.sub('\.', '.\n', text)
    with open(fileName, 'a') as file:
        file.write(formatText)
        choice = input("Do you want to translate audio transcription to English? (Yes/No) ")
    if (choice == "Yes"):
        translateResult('transcription.txt', 'translation.txt', formatText)

def getResult():
	# Select speech recognition model
    modelName = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    model = whisper.load_model(modelName,device=checkDevice())
    # Save transcribed text to file
    result = model.transcribe(audiofile)
    formatResult('transcription.txt', result["text"])

def main():
    getAudio() # Download an mp3 audio file to transcribe to text
    getResult() # Get transcription


if __name__ == "__main__":
    main()
