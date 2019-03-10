from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import flask
from multiprocessing.pool import ThreadPool
import numpy as np
import socket
import cv2
import math
import os
import tensorflow as tf
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
from timeit import default_timer as timer
import io
from pydub import AudioSegment

from text_translation_Azure_1 import translate
from text_to_speech_Azure_2 import texttospeech


# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
g = None
model = None
restore_fn = None
vocab = None
pool = ThreadPool()

# Ps: the localhost can be exposed to the world with ngrok "./ngrok http 5000"

class Server(object):
    """
    Server
    """

    def __init__(self):
        print("Server ... starting server and loading model ... please wait until its started ...")

        ai_periscope_path = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope"
        checkpoint_path = ai_periscope_path+"/im2txt_models/Pretrained-Show-and-Tell-model/model.ckpt-2000000"
        vocab_file = ai_periscope_path+"/im2txt_models/Pretrained-Show-and-Tell-model/word_counts.txt"

        ## nope these versions are sadly not working...
        ##checkpoint_path = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/im2txt_models/upload/model_conv.ckpt-3000000"
        ##vocab_file = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/im2txt_models/upload/word_counts.txt"

        self.load_model_im2txt(checkpoint_path, vocab_file)

        """
        from cStringIO import StringIO
        buf = StringIO()
        response = flask.make_response(buf.getvalue())
        buf.close()
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
        return response
        """

    def load_model_im2txt(self, checkpoint_path, vocab_file):
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
                    str_formating = sentence
                    if str_formating[-2:] == " .":
                        str_formating = str_formating[:-2] + "."

                    str_formating = str_formating.capitalize()
                    response += "%s" % (str_formating)

                    break

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

                    str_formating = sentence
                    if str_formating[-2:] == " .":
                        str_formating = str_formating[:-2]+"."

                    str_formating = str_formating.capitalize()
                    response += "%s" % (str_formating)

                    break

        return response

@app.route('/speech.wav', methods=['GET', 'POST'])
def speechwav():
    filename = "speech.wav"
    return flask.send_from_directory(directory="", filename=filename)
@app.route('/speech.mp3', methods=['GET', 'POST'])
def speechmp3():
    filename = "speech.mp3"
    return flask.send_from_directory(directory="", filename=filename)

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

### PYTHON CLIENT >>>>
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/python_binding", methods=["GET","POST"])
def python_binding():
    start = timer()

    data = {"success": False}
    if flask.request.method == "POST":
        uids = []
        imgs_data = []

        print("flask.request.files",flask.request.files)
        print("flask.request.values", flask.request.values)

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

        # indicate that the request was a success
        data["success"] = True

    image = images[0]
    lang_code = flask.request.values["language"]

    print(image.shape)
    print("lang_code", lang_code)

    str_description = ""
    try:
        # resize?
        cv2.imwrite("tmp.jpg",image) ## hahahaaack

        input_files = "tmp.jpg"
        images = []
        with tf.gfile.GFile(input_files, "rb") as f:
            image = f.read()
        images.append(image)

        str_description = server.run_model_on_images(images)
        success = True

    except Exception as e:
        print("Exception caught!!!", e)


    # language stuffs
    if lang_code is not "en":
        translated = translate(str_description, lang_code)

        print("Text translation >", translated)

        translated_text = (translated[0]["translations"][0]["text"])
        print(translated_text)
        str_description = translated_text

    print("Saving audio")
    texttospeech(str_description, lang_code)

    wav_filepath = "speech.wav"
    mp3_filepath = "speech.mp3"
    sound = AudioSegment.from_wav(wav_filepath)
    sound.export(mp3_filepath, format="mp3")

    #return flask.jsonify(data)
    #mp3_filepath = "mini.mp3"
    #response = flask.make_response(flask.send_file(mp3_filepath))
    #response.headers['Location'] = str ### haaaaaack
    #return response

    end = timer()
    time = (end - start)

    data = {"success": success, "str": str_description, "internal_time": time}

    response = flask.make_response(flask.send_file(mp3_filepath))
    response.headers['Location'] = str_description ### haaaaaack

    return flask.jsonify(data)
    return response



    data = {"success": False}


    start = timer()

    input_files = "/home/vitek/Vitek/Projects_local_for_ubuntu/AI_Periscope/models/im2txt/im2txt/img3.jpg"
    data = {"success": True}
    data['language'] = "eng"
    data["internal_time"] = 0.01337080085
    str = server.run_model_on_image_paths(input_files)
    data["str"] = str

    lang_code = "zh" #"de"

    # language stuffs
    if lang_code is not "en":
        translated = translate(str, lang_code)

        print("Text translation >", translated)

        translated_text = (translated[0]["translations"][0]["text"])
        print(translated_text)

        print("Saving audio")
        texttospeech(translated_text, lang_code)

        str = translated_text

    wav_filepath = "speech.wav"
    mp3_filepath = "speech.mp3"
    sound = AudioSegment.from_wav(wav_filepath)
    sound.export(mp3_filepath, format="mp3")

    #return flask.jsonify(data)
    end = timer()

    response = flask.make_response(flask.send_file(mp3_filepath))
    response.headers['Location'] = str ### haaaaaack

    time_spent = end - start
    print("Evaluation took: ", time_spent)

    return response


### FIRST VERSION >>>>

@app.route("/evaluate_image_local", methods=["GET","POST"])
def evaluate_image_local():
    input_files = "/media/vitek/VitekDrive_I/2019_Projects/AI_Periscope/images/img3.jpg"
    input_files = "/home/vitek/Vitek/Projects_local_for_ubuntu/AI_Periscope/models/im2txt/im2txt/img3.jpg"
    data = {"success": True}
    data['language'] = "eng"
    data["internal_time"] = 0.01337080085
    str = server.run_model_on_image_paths(input_files)
    data["str"] = str

    return flask.jsonify(data)

@app.route("/demo_stuffs", methods=["GET","POST"])
def demo_stuffs():
    start = timer()

    input_files = "/home/vitek/Vitek/Projects_local_for_ubuntu/AI_Periscope/models/im2txt/im2txt/img3.jpg"
    data = {"success": True}
    data['language'] = "eng"
    data["internal_time"] = 0.01337080085
    str = server.run_model_on_image_paths(input_files)
    data["str"] = str

    lang_code = "zh" #"de"

    # language stuffs
    if lang_code is not "en":
        translated = translate(str, lang_code)

        print("Text translation >", translated)

        translated_text = (translated[0]["translations"][0]["text"])
        print(translated_text)

        print("Saving audio")
        texttospeech(translated_text, lang_code)

        str = translated_text

    wav_filepath = "speech.wav"
    mp3_filepath = "speech.mp3"
    sound = AudioSegment.from_wav(wav_filepath)
    sound.export(mp3_filepath, format="mp3")

    #return flask.jsonify(data)
    end = timer()

    response = flask.make_response(flask.send_file(mp3_filepath))
    response.headers['Location'] = str ### haaaaaack

    time_spent = end - start
    print("Evaluation took: ", time_spent)

    return response


@app.route("/debug", methods=["GET","POST"])
def debug():
    print("DEBUG")
    print("values", flask.request.values)
    print("args", flask.request.args)
    print("form", flask.request.form)
    print("data", flask.request.data)
    print("files", flask.request.files)

    """
    file = flask.request.files['file']

    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    print("Received image:", img.shape)
    """
    data = {"success": True}

    return flask.jsonify(data)

### FIRST VERSION >>>>

@app.route("/eval_str", methods=["GET","POST"])
def eval_str():
    str = ""
    start = timer()
    success = False
    #data = {"3":"eng"}

    print("Handshake!")
    vals = flask.request.values
    print("values: ", vals)
    form = flask.request.form
    print("form: ", form)

    """
    lang_code = "en"
    try:
        # awfully hacky ~-~-~-~

        d = flask.request.form.to_dict()
        #print("d",d)
        #d {'Language': '{"selectedlanguage":"Mandarin"}'}

        lang = d['Language']
        print("Selected language:", lang)
        #lang_selected = lang['selectedlanguage']
        #{"selectedlanguage":"English"}
        lang_selected = lang[21:-2] # is str now...
        print("Selected language:", lang_selected)

        if lang_selected == "ul":
            print("Is NULL!")
            lang_selected = "English"
        # "German", "Mandarin", "English"

        # language to Microsoft code
        if lang_selected == "German":
            lang_code = "de"
        elif lang_selected == "Mandarin":
            lang_code = "zh"

    except Exception as e:
        print("Exception caught reading language...", e)
    """

    try:

        file = flask.request.files['file']

        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)

        # resize?
        cv2.imwrite("tmp.jpg",img) ## hahahaaack

        input_files = "tmp.jpg"
        images = []
        with tf.gfile.GFile(input_files, "rb") as f:
            image = f.read()
        images.append(image)

        str = server.run_model_on_images(images)
        success = True

    except Exception as e:
        print("Exception caught!!!", e)

    """
    # language stuffs
    if lang_code is not "en":
        translated = translate(str, lang_code)

        print("Text translation >", translated)

        translated_text = (translated[0]["translations"][0]["text"])
        print(translated_text)

        print("Saving audio")
        texttospeech(translated_text, lang_code)

        str = translated_text

    wav_filepath = "speech.wav"
    mp3_filepath = "speech.mp3"
    sound = AudioSegment.from_wav(wav_filepath)
    sound.export(mp3_filepath, format="mp3")

    #return flask.jsonify(data)
    """
    end = timer()

    mp3_filepath = "mini.mp3"
    response = flask.make_response(flask.send_file(mp3_filepath))
    response.headers['Location'] = str ### haaaaaack

    return response


@app.route("/eval", methods=["GET","POST"])
def eval():
    str = ""
    start = timer()
    success = False
    #data = {"3":"eng"}

    print("Handshake!")
    vals = flask.request.values
    print("values: ", vals)
    form = flask.request.form
    print("form: ", form)

    """
    values: CombinedMultiDict([ImmutableMultiDict([]), ImmutableMultiDict([('language', 'eng')])])
    args: ImmutableMultiDict([])
    form: ImmutableMultiDict([('language', 'eng')])
    data: b''
    """

    lang_code = "en"
    try:
        # awfully hacky ~-~-~-~

        d = flask.request.form.to_dict()
        #print("d",d)
        #d {'Language': '{"selectedlanguage":"Mandarin"}'}

        lang = d['Language']
        print("Selected language:", lang)
        #lang_selected = lang['selectedlanguage']
        #{"selectedlanguage":"English"}
        lang_selected = lang[21:-2] # is str now...
        print("Selected language:", lang_selected)

        if lang_selected == "ul":
            print("Is NULL!")
            lang_selected = "English"
        # "German", "Mandarin", "English"

        # language to Microsoft code
        if lang_selected == "German":
            lang_code = "de"
        elif lang_selected == "Mandarin":
            lang_code = "zh"

    except Exception as e:
        print("Exception caught reading language...", e)

    try:

        file = flask.request.files['file']

        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)

        # resize?
        cv2.imwrite("tmp.jpg",img) ## hahahaaack

        input_files = "tmp.jpg"
        images = []
        with tf.gfile.GFile(input_files, "rb") as f:
            image = f.read()
        images.append(image)

        str = server.run_model_on_images(images)
        success = True

    except Exception as e:
        print("Exception caught!!!", e)


    # language stuffs
    if lang_code is not "en":
        translated = translate(str, lang_code)

        print("Text translation >", translated)

        translated_text = (translated[0]["translations"][0]["text"])
        print(translated_text)

        print("Saving audio")
        texttospeech(translated_text, lang_code)

        str = translated_text

    wav_filepath = "speech.wav"
    mp3_filepath = "speech.mp3"
    sound = AudioSegment.from_wav(wav_filepath)
    sound.export(mp3_filepath, format="mp3")

    #return flask.jsonify(data)
    end = timer()

    response = flask.make_response(flask.send_file(mp3_filepath))
    response.headers['Location'] = str ### haaaaaack

    return response

    import json
    response = app.response_class(response=json.dumps(data),
                                  status=200,
                                  mimetype='application/json')
    return response


    #data['language'] = "eng"
    #data["success"] = success
    #data["internal_time"] = end - start
    #data["str"] = str
    data[3] = "eng"
    data[1] = success
    data[4] = end - start
    data[2] = str

    print("got all the way before jasonify")
    r = flask.jsonify(data)
    print("after it")
    # return the data dictionary as a JSON response
    return r


@app.route("/evaluate_image", methods=["GET","POST"])
def evaluate_image():
    str = ""
    start = timer()

    data = {"success": False}
    print("Handshake!")
    try:
        #print("values",flask.request.values)
        #print("files",flask.request.files)
        language = flask.request.values['language']
        data['language'] = language

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

        img = images[0]
        cv2.imwrite("tmp.jpg",img) ## hahahaaack

        input_files = "tmp.jpg"
        images = []
        with tf.gfile.GFile(input_files, "rb") as f:
            image = f.read()
        images.append(image)
        #print(type(images), type(images[0]))
        #print(images)


        str = server.run_model_on_images(images)
        data["success"] = True

    except Exception as e:
        print("Exception caught!!!", e)

    end = timer()

    data["internal_time"] = end - start
    data["str"] = str

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

    return str

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