#!/usr/bin/env python3
"""
Audio transcriber using OpenAI's Whisper speech recognition model.
Usage: python3 transcriber.py -u, --url <URL>
"""
import getopt
import re
import sys
import torch
import whisper

from googletrans import Translator
import yt_dlp as youtube_dl


AUDIOFILE = "audio.mp3"  # Save audio file as audio.mp3


def match_pattern(pattern, arg):
    """If YouTube shorts URL is given, convert it to standard URL."""
    match = re.search(pattern, arg)
    if bool(match):
        url = re.sub(pattern, "watch?v=", arg)
    else:
        url = arg
    return url


def get_audio(url, argv):
    """
    Download mp3 audio of a YouTube video. Credit to Stokry.
    https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
    """
    try:
        opts, args = getopt.getopt(argv, "u:", ["url="])
    except:
        print("Usage: python3 transcriber.py -u <url>")
    for opt, arg in opts:
        if opt in ['-u', '--url']:
            url = match_pattern("shorts/", arg)
    video_info = youtube_dl.YoutubeDL().extract_info(url=url, download=False)
    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': AUDIOFILE,
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])


def banner(text):
    """Display a message when the script is working in the background"""
    print(f"# {text} #")


def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_result():
    """Get speech recognition model."""
    model_name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    banner("Transcribing text")
    model = whisper.load_model(model_name, device=check_device())
    result = model.transcribe(AUDIOFILE)
    format_result('transcription.txt', result["text"])


def format_result(file_name, text):
    """Put a newline character after each sentence and prompt user for translation."""
    format_text = re.sub('\.', '.\n', text)
    with open(file_name, 'a', encoding="utf-8") as file:
        banner("Writing transcription to text file")
        file.write(format_text)
        choice = input("Do you want to translate audio transcription to English? (Yes/No) ")
    if choice == "Yes":
        translate_result('transcription.txt', 'translation.txt')


def translate_result(org_file, trans_file):
    """
    Translate transcribed text. Credit to Harsh Jain at educative.io
    https://www.educative.io/answers/how-do-you-translate-text-using-python
    """
    translator = Translator()  # Create an instance of Translator() class
    with open(org_file, 'r', encoding="utf-8") as transcription:
        contents = transcription.read()
        banner("Translating text")
        translation = translator.translate(contents)
    with open(trans_file, 'a', encoding="utf-8") as file:
        banner("Writing translation to text file")
        file.write(translation.text)


def main():
    """Main function."""
    get_audio(None,sys.argv[1:])    # Download an mp3 audio file to transcribe to text
    get_result()            # Get audio transcription and translation if needed

if __name__ == "__main__":
    main()
