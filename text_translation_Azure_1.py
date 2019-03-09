# -*- coding: utf-8 -*-
import os, requests, uuid, json


def translate(output, language):
    lang = ('&to=%s' % language)

    # Checks to see if the Translator Text subscription key is available
    # as an environment variable. If you are setting your subscription key as a
    # string, then comment these lines out.

    # If you want to set your subscription key as a string, uncomment the line
    # below and add your subscription key.
    subscriptionKey = '41acfe5211114bb58080d20504d7d447'

    base_url = 'https://api-eur.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = lang
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscriptionKey,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': output
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    b = response
    a = (json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))
    return b