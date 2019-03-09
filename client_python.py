import cv2
import numpy as np
import datetime, time
from threading import Thread
import requests
from timeit import default_timer as timer

class video_capture(object):
    """
    Generates individual frames from a stream of connected camera.
    Provides image when asked for.
    """

    def __init__(self, src=0):
        self.capture_handle = cv2.VideoCapture(src)

    def get_frame(self):
        time_start = timer()

        ret, frame = self.capture_handle.read()

        time_end = timer()
        fps = 1.0 / (time_end - time_start)
        return ret, frame, fps

    def destroy(self):
        self.capture_handle.release()
        cv2.destroyAllWindows()


class renderer(object):
    """
    Draw image to screen.
    """

    def __init__(self, video_capture):
        # Init stuff

        self.video_capture = video_capture
        self.sample_every = 1 # sec


        return None

    def run_on_frames_everytick(self, show=True, tick=1):
        self.sample_every = tick

        time_start = timer()
        while (True):
            key = cv2.waitKey(1)
            if key == ord('1'):
                break
            if key == ord('9'):
                self.sample_every /= 2.0
            if key == ord('0'):
                self.sample_every *= 2.0

            ret, frame, fps = self.video_capture.get_frame()

            time_now = timer()
            if (time_now - time_start) > self.sample_every:
                time_start = time_now

                timestamp = time.time()
                time_value = datetime.datetime.fromtimestamp(timestamp)
                print(time_value.strftime('%Y-%m-%d %H:%M:%S'))

                # HERE DO SOMETHING WITH THE IMAGE (every self.sample_every sec)
                a = self.do_cool_beans_here(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(frame, "FPS "+'{:.2f}'.format(fps)+", sample rate "+'{:.3f}'.format(self.sample_every), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if show:
                cv2.imshow('frame', frame)

    def do_cool_beans_here(self, frame):
        PORT = "5000"
        PRED_KERAS_REST_API_URL = "http://localhost:" + PORT + "/python_binding"
        values = {"language": "zh"} # en, de, zh

        print (frame.shape)

        images = [frame]
        number_of_images = len(images)

        payload = {}

        for i in range(number_of_images):
            image = images[i]
            image_enc = cv2.imencode('.jpg', image)[1].tostring()
            payload[str(0)] = image_enc

        try:
            r = requests.post(PRED_KERAS_REST_API_URL, files=payload, data=values).json()
        except Exception as e:
            print("CONNECTION TO SERVER ",PRED_KERAS_REST_API_URL," FAILED - return to backup local evaluation?")
            print("Exception:", e)

        print("request data", r)
        return True


video_capture = video_capture()
renderer = renderer(video_capture)

#renderer.show_frames()

tick = 5 # every ten seconds!

renderer.run_on_frames_everytick(show=True,tick=tick)
#renderer.runopenface_on_frames_nowaiting(show=True)
#renderer.record_frames()


video_capture.destroy()
