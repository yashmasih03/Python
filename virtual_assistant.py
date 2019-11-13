# Description: This is a virtual assistant program that gets the date, responds back with a random greeting, and returns a response
#             getting information on a person from wikipedia e.g. 'who is LeBron James'

# A virtual assistant is an application that can understand voice commands and complete tasks for a user.

# Resources:
# (1) https://towardsdatascience.com/build-your-first-voice-assistant-85a5a49f6cc1
# (2) https://pythonspot.com/personal-assistant-jarvis-in-python/
# (3) https://realpython.com/python-speech-recognition/
# (4) https://pypi.org/project/SpeechRecognition/1.2.3/
# (5) https://stackabuse.com/getting-started-with-pythons-wikipedia-api/


# pip install pyaudio
# pip install SpeechRecognition
# pip install gTTS
# pip install wikipedia

# Import the libraries
import speech_recognition as sr
import os
from gtts import gTTS
import datetime
import warnings
import calendar
import random
import wikipedia

# Ignore any warning messages
warnings.filterwarnings('ignore')


# Record audio and return it as a string
def recordAudio():
    # Record the audio
    r = sr.Recognizer()
    with sr.Microphone() as source:  # The with statement itself ensures proper acquisition and release of resources
        print('Say something!')
        audio = r.listen(source)

    # Speech recognition using Google's Speech Recognition
    data = ''
    try:
        data = r.recognize_google(audio)
        print('You said: ' + data)
    except sr.UnknownValueError:
        print('Google Speech Recognition could not understand the audio,  unknown error')
    except sr.RequestError as e:
        print('Request results from Google Speech Recognition service error ' + e)

    return data


# Function to get the virtual assistant response
def assistantResponse(text):
    print(text)

    # Convert the text to speech
    myobj = gTTS(text=text, lang='en', slow=False)

    # Save the converted audio to a file
    myobj.save('assistant_response.mp3')

    # Play the converted file
    os.system('start assistant_response.mp3')


# A function to check for wake word(s)
def wakeWord(text):
    WAKE_WORDS = ['hey computer', 'okay computer'] # A list of wake words

    text = text.lower()  # Convert the text to all lower case words
    # Check to see if the users command/text contains a wake word
    for phrase in WAKE_WORDS:
        if phrase in text:
            return True

    # If the wake word wasn't found in the loop then return False
    return False


def getDate():
    now = datetime.datetime.now()
    my_date = datetime.datetime.today()
    weekday = calendar.day_name[my_date.weekday()]  # e.g. Monday
    monthNum = now.month
    dayNum = now.day

    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                   'October', 'November', 'December']
    ordinalNumbers = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th',
                      '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd',
                      '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']
    return 'Today is ' + weekday + ' ' + month_names[monthNum - 1] + ' the ' + ordinalNumbers[dayNum - 1] + '.'


# Function to return a random greeting response
def greeting(text):
    # Greeting Inputs
    GREETING_INPUTS = ['hi', 'hey', 'hola', 'greetings', 'wassup', 'hello']

    # Greeting Response back to the user
    GREETING_RESPONSES = ['howdy', 'whats good', 'hello', 'hey there']

    # if the users input is a greeting, then return a randomly chosen greeting response
    for word in text.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES) + '.'

    # If no greeting was detected then return an empty string
    return ''

# function to get information on a person
def getPerson(text):
    wordList = text.split()  # Split the text into a list of words

    for i in range(0, len(wordList)):
        if i + 3 <= len(wordList) - 1 and wordList[i].lower() == 'who' and wordList[i + 1].lower() == 'is':
            return wordList[i + 2] + ' ' + wordList[i + 3]


while True:
    # Record the audio
    text = recordAudio()
    response = '' #Empty response string to append the text of the virtual assistants response

    # Checking for the wake word/phrase
    if (wakeWord(text) == True):

        # Check for greetings by the user
        response = response + greeting(text) 

        #Check to see if the user said anything having to do with the date
        if ('date' in text):
            get_date = getDate()
            response = response + ' ' + get_date

        #Check to see if the user said anything having to do with the time
        if('time' in text):
            now = datetime.datetime.now()
            meridiem = ''
            if now.hour >= 12:
                meridiem = 'p.m' #Post Meridiem (PM)
                hour = now.hour - 12
            else:
                meridiem = 'a.m'#After Meridiem (AM)
            response = response + ' '+'It is '+str(hour) +':'+ str(now.minute)+' '+meridiem+' .'
        
        #Check to see if the user said 'who is'
        if ('who is' in text):
            person = getPerson(text)
            wiki = wikipedia.summary(person, sentences=2)
            response = response + ' ' + wiki

       # Have the assistant respond  back using audio and the text from response
        assistantResponse(response)
