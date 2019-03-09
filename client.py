import requests
from timeit import default_timer as timer

PORT = "5000"
PRED_KERAS_REST_API_URL = "http://localhost:"+PORT+"/evaluate_image"
TIME_KERAS_REST_API_URL = "http://localhost:"+PORT+"/handshake"
IMAGE_PATH = "img.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
start = timer()
r = requests.post(TIME_KERAS_REST_API_URL, files=payload).json()
end = timer()
print("Time", (end-start))
print("request data", r)

start = timer()
r = requests.post(PRED_KERAS_REST_API_URL, files=payload).json()
end = timer()
print("Time", (end-start))
print("request data", r)