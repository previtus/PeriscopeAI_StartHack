from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread
import time

from PIL import Image
import flask
import io
import os
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool
import numpy as np
import socket
import cv2


import math
import os


import tensorflow as tf
#os.chdir("/home/vitek/Vitek/Projects_local_for_ubuntu/AI_Periscope/models/im2txt/im2txt")
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

#from im2txt import configuration
#from im2txt import inference_wrapper
#from im2txt.inference_utils import caption_generator
#from im2txt.inference_utils import vocabulary

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
g = None
model = None
restore_fn = None
vocab = None
pool = ThreadPool()

# del
from timeit import default_timer as timer
import numpy

times_del = []

class Server(object):
    """
    Server
    """

    def __init__(self):
        print("Server ... starting server and loading model ... please wait until its started ...")
        self.warm_up = 0

        ai_periscope_path = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope"
        checkpoint_path = ai_periscope_path+"/im2txt_models/Pretrained-Show-and-Tell-model/model.ckpt-2000000"
        vocab_file = ai_periscope_path+"/im2txt_models/Pretrained-Show-and-Tell-model/word_counts.txt"

        input_files = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/images/img3.jpg"
        self.load_model_im2txt(checkpoint_path, vocab_file, input_files)

        #print("1")
        #self.run_model_on_image_paths(input_files)
        #print("2")
        #self.run_model_on_image(input_files)

        """
        frequency_sec = 10.0
        t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
        t.daemon = True
        t.start()

        # hack to distinguish server
        # this might not work on non gpu machines
        # but we are using only those
        hostname = socket.gethostname()  # gpu048.etcetcetc.edu
        if hostname[0:3] == "gpu":
            app.run(host='0.0.0.0', port=8123)
        else:
            app.run()
        """

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_model_im2txt(self, checkpoint_path, vocab_file, input_files):
        global g
        global model
        global restore_fn
        global vocab

        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                       checkpoint_path)
        g.finalize()

        # Create the vocabulary.
        vocab = vocabulary.Vocabulary(vocab_file)

        print('Model loaded.')

    def run_model_on_image_paths(self, input_files_paths):
        response = ""

        filenames = []
        for file_pattern in input_files_paths.split(","):
            filenames.extend(tf.gfile.Glob(file_pattern))
        tf.logging.info("Running caption generation on %d files matching %s",
                        len(filenames), input_files_paths)

        with tf.Session(graph=g) as sess:
            # Load the model from checkpoint.
            restore_fn(sess)

            # Prepare the caption generator. Here we are implicitly using the default
            # beam search parameters. See caption_generator.py for a description of the
            # available beam search parameters.
            generator = caption_generator.CaptionGenerator(model, vocab)

            for filename in filenames:
                with tf.gfile.GFile(filename, "rb") as f:
                    image = f.read()
                captions = generator.beam_search(sess, image)
                print("Captions for image %s:" % os.path.basename(filename))
                for i, caption in enumerate(captions):
                    # Ignore begin and end words.
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                    response += "  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob))

        return response

    def run_model_on_images(self, images):
        response = ""
        with tf.Session(graph=g) as sess:
            # Load the model from checkpoint.
            restore_fn(sess)

            generator = caption_generator.CaptionGenerator(model, vocab)

            for i,image in enumerate(images):
                captions = generator.beam_search(sess, image)
                print("Captions for image:", i)
                for i, caption in enumerate(captions):
                    # Ignore begin and end words.
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                    response += "  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob))

        return response


@app.route("/handshake", methods=["GET","POST"])
def handshake():
    # Handshake
    start = timer()

    data = {"success": False}
    print("Handshake!")
    try:
        hostname = socket.gethostname()  # gpu048.etcetcetc.edu
        machine_name = hostname.split(".")[0]
        buses = get_gpus_buses()
        print("Bus information =", buses)
        if len(buses) > 0:
            buses = ":" + buses
        data["server_name"] = machine_name + buses
    except Exception as e:
        data["server_name"] = "Bob!"

    end = timer()

    data["internal_time"] = end - start
    data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/evaluate_image_local", methods=["GET","POST"])
def evaluate_image_local():
    input_files = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/images/img3.jpg"
    str = server.run_model_on_image_paths(input_files)
    return str


@app.route("/evaluate_image", methods=["GET","POST"])
def evaluate_image():
    input_files = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/images/img3.jpg"

    images = []
    with tf.gfile.GFile(input_files, "rb") as f:
        image = f.read()
    images.append(image)

    str = ""
    start = timer()

    data = {"success": False}
    print("Handshake!")
    try:
        str = server.run_model_on_images(images)
        data["success"] = True

    except Exception as e:
        print("Exception caught!!!")

    end = timer()

    data["internal_time"] = end - start
    data["str"] = str

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

    return str

    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        uids = []
        imgs_data = []

        t_start_decode = timer()
        for key in flask.request.files:
            im_data = flask.request.files[key].read()

            imgs_data.append(im_data)
            uids.append(key)

        images = pool.map(lambda i: (
            cv2.imdecode(np.asarray(bytearray(i), dtype=np.uint8), 1)
        ), imgs_data)

        t_start_eval = timer()
        print("Received",len(images),"images (Decoded in",(t_start_eval-t_start_decode),".", uids, [i.shape for i in images])

        str = server.run_model_on_images(images)
        t_end_eval = timer()

        data["bboxes"] = results_bboxes
        data["uids"] = uids
        data["time_pure_eval"] = t_end_eval-t_start_eval
        data["time_pure_decode"] = t_start_eval-t_start_decode

        # indicate that the request was a success
        data["success"] = True

    return flask.jsonify(data)

def my_img_to_array(img):
    # remove Keras dep
    x = np.asarray(img, dtype='float32')
    return x

from tensorflow.python.client import device_lib
def get_gpus_buses():
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
    buses = ""
    for device in gpu_devices:
        desc = device.physical_device_desc # device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:81:00.0
        bus = desc.split(",")[-1].split(" ")[-1][5:] # split to get to the bus information
        bus = bus[0:2] # idk if this covers every aspect of gpu bus
        if len(buses)>0:
            buses += ";"
        buses += str(bus)
    return buses


#if __name__ == "__main__":
server = Server()