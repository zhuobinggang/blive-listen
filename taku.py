import pyttsx3
engine = pyttsx3.init()
engine.setProperty('voice','zh')

def say(text):
    engine.say(text)
    engine.runAndWait()

def test():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        print(voice.id)
        print(voice.name)
        print(voice.languages)
        print(voice.gender)
        print(voice.age)

def black(text):
    print('\033[30m', text, '\033[0m', sep='')

def red(text):
    print('\033[31m', text, '\033[0m', sep='')

def green(text):
    print('\033[32m', text, '\033[0m', sep='')

def yellow(text):
    print('\033[33m', text, '\033[0m', sep='')

def blue(text):
    print('\033[34m', text, '\033[0m', sep='')

def magenta(text):
    print('\033[35m', text, '\033[0m', sep='')

def cyan(text):
    print('\033[36m', text, '\033[0m', sep='')

def gray(text):
    print('\033[90m', text, '\033[0m', sep='')


def color_print(text):
    magenta(text)
