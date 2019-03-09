import sys
import requests
from timeit import default_timer as timer

PORT = "5000"
PRED_KERAS_REST_API_URL = "http://localhost:"+PORT+"/evaluate_image"
TIME_KERAS_REST_API_URL = "http://localhost:"+PORT+"/handshake"
DEBUG_KERAS_REST_API_URL = "http://localhost:"+PORT+"/debug"

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = "img.jpg"

if len(sys.argv) > 2:
    lang = sys.argv[2]
else:
    lang = "eng"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
values = {'language': lang}

start = timer()
r = requests.post(DEBUG_KERAS_REST_API_URL, data=values).json()
end = timer()
print("Time", (end-start))
print("request data", r)


"""
# submit the request
start = timer()
r = requests.post(TIME_KERAS_REST_API_URL, files=payload).json()
end = timer()
print("Time", (end-start))
print("request data", r)


start = timer()
r = requests.post(PRED_KERAS_REST_API_URL, files=payload, data=values).json()
end = timer()
print("Time", (end-start))
print("request data", r)


### show

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread(IMAGE_PATH)
imgplot = plt.imshow(img)
plt.show()

"""