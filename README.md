# Running backend:

Using these models https://github.com/KranthiGV/Pretrained-Show-and-Tell-model

And a c22611891d0826bbf656a367874489c0dad95777 commit of im2txt from tensorflow models

## instructions:

Start server:

FLASK_APP=server_runner.py flask run

- for experiments outside of localhost - useful application for exposing your port to the outside world - ngrok :

./ngrok http 5000

Run python client:

python client_python.py

or use the app which also connects to the same server



## installed:

Package              Version  
-------------------- ---------
absl-py              0.7.0    
astor                0.7.1    
certifi              2018.8.24
Click                7.0      
Flask                1.0.2    
gast                 0.2.2    
grpcio               1.19.0   
h5py                 2.9.0    
itsdangerous         1.1.0    
Jinja2               2.10     
Keras-Applications   1.0.7    
Keras-Preprocessing  1.0.9    
Markdown             3.0.1    
MarkupSafe           1.1.1    
mock                 2.0.0    
nltk                 3.4      
numpy                1.16.2   
opencv-python        4.0.0.21 
pbr                  5.1.3    
Pillow               5.4.1    
pip                  19.0.3   
protobuf             3.7.0    
setuptools           40.2.0   
singledispatch       3.4.0.3  
six                  1.12.0   
tensorboard          1.13.1   
tensorflow           1.1.0    
tensorflow-estimator 1.13.0   
termcolor            1.1.0    
Werkzeug             0.14.1   
wheel                0.31.1   

