# Python program to translate
# speech to text and text to speech

import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
import Testing

r = sr.Recognizer()

engine = pyttsx3.init()
newVoiceRate = 150
engine.setProperty('rate',newVoiceRate)
voice = engine.getProperty('voices')
engine.setProperty('voice', voice[1].id)
# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    print(command)
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to
# speak
flag = True
SpeakText("Hi there! I am Chat bot, what are you looking for ...... \n")
i = 0
while (1):

    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=1.0)


            print()
            # listens for the user's input
            print("Let's say some thing OK ^^! (^_^))")
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            message = MyText
            # print(MyText)
            ints = Testing.predict_class(message)
            res, flag  = Testing.getResponse(ints,Testing.intents)

            if flag == False:
                SpeakText(res)
                break

                # Check if the 'voice' list is not empty and the voice index is within range
            if voice and i < len(voice):
                engine.setProperty('voice', voice[i].id)
                i += 2
            else:
                # Reset the voice index to 0 if it exceeds the list length or if the list is empty
                i = 0

            SpeakText(res)
            SpeakText("Is there anything I can help you?")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        SpeakText("I am listening to your question!")