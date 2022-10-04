#!/usr/bin/env python3

import youtube_dl
import subprocess
import whisper
import sys

# Download mp3 audio of a Youtube video. Credit to Stokry
# https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
def audio():
    video_url = input("Enter Youtube URL: ")
    video_info = youtube_dl.YoutubeDL().extract_info(url = video_url,download=False)
    filename = "audio.mp3"
    options={
        'format':'bestaudio/best',
        'keepvideo':False,
        'outtmpl':filename,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

def main():
    audio() # Download an mp3 audio file to transcribe to text
    # Select speech recognition model
    model = input("Select speech recognition model (tiny, base, small, medium, large): ")
    model = whisper.load_model(model)
    result = model.transcribe("audio.mp3")
    file = open('transcription.txt', 'w')
    file.write(result["text"])
    file.write("\n")
    file.close

if __name__ == "__main__":
    main()
