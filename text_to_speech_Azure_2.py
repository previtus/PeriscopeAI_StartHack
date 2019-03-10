import os, requests, time
from xml.etree import ElementTree

def texttospeech (text,language):
    voice2s= '1'
    if language =='en':
        voice2s= 'en-us'
        person='Guy24KRUS'
    if language =="de":
        voice2s='de-DE'
        person='Hedda'
    if language =="zh":
        voice2s = 'zh-CN'
        person = 'HuihuiRUS'
    if language =="es":
        voice2s = 'es-ES'
        person = 'Laura, Apollo'
    if language =="fr":
        voice2s = 'fr-CH'
        person = 'Guillaume'
    if language =="hi": #hindi
        voice2s = 'hi-IN'
        person = 'Kalpana, Apollo'
    if language =="cs":
        voice2s = 'cs-CZ'
        person = 'Jakub'
    if language =="ms": # malay
        voice2s = 'ms-MY'
        person = 'Rizwan'
    if language =="it":
        voice2s = 'it-IT'
        person = 'Cosimo, Apollo'
    if language =="ja":
        voice2s = 'ja-JP'
        person = 'Ichiro, Apollo'

    class TextToSpeech(object):
        def __init__(self, subscription_key):
            self.subscription_key = subscription_key
            self.tts = text
            self.timestr = time.strftime("%Y%m%d-%H%M")
            self.access_token = None

        def get_token(self):
            fetch_token_url = "https://westeurope.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
            headers = {
                'Ocp-Apim-Subscription-Key': self.subscription_key
            }
            response = requests.post(fetch_token_url, headers=headers)
            self.access_token = str(response.text)

        def save_audio(self):
            base_url = 'https://westeurope.tts.speech.microsoft.com/'
            path = 'cognitiveservices/v1'
            constructed_url = base_url + path
            headers = {
                'Authorization': 'Bearer ' + self.access_token,
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
                'User-Agent': 'YOUR_RESOURCE_NAME'
            }
            xml_body = ElementTree.Element('speak', version='1.0')
            xml_body.set('{http://www.w3.org/XML/1998/namespace}lang', voice2s)
            voice = ElementTree.SubElement(xml_body, 'voice')
            voice.set('{http://www.w3.org/XML/1998/namespace}lang', voice2s)
            voice.set('name', 'Microsoft Server Speech Text to Speech Voice (%s, %s)'%(voice2s, person))
            voice.text = self.tts
            body = ElementTree.tostring(xml_body)

            response = requests.post(constructed_url, headers=headers, data=body)
            if response.status_code == 200:
                with open('speech.wav', 'wb') as audio:
                    audio.write(response.content)
                    print("\nStatus code: " + str(response.status_code) + "\nYour TTS is ready for playback.\n")
            else:
                print("\nStatus code: " + str(response.status_code) + "\nSomething went wrong. Check your subscription key and headers.\n")

            return response.status_code


    #if __name__ == "__main__":
    subscription_key = "ENTER YOUR API KEY HERE"
    app = TextToSpeech(subscription_key)
    app.get_token()
    success = app.save_audio()
    print(success)


