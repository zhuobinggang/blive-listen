import pyttsx3
engine = pyttsx3.init()

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

