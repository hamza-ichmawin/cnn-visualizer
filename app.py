from flask import Flask, render_template, request, jsonify
from keras.layers.convolutional.conv2d import Conv2D
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os
import shutil
import base64
import io

app = Flask(__name__)


UPLOAD_FOLDER = 'models/Current'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
    model_file = request.files.get('model_file')
    if model_file:
        # Save the model file to the specified directory
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        model_filename = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
        model_file.save(model_filename)
        folder_model_user = os.path.join("models/stolen models/", model_file.filename.split(".")[0])
        if not os.path.exists(folder_model_user):
            os.mkdir(folder_model_user)
            shutil.copy(model_filename, os.path.join(folder_model_user,model_file.filename))
        return f"'{model_file.filename}' uploaded successfully"

    return "No file uploaded"


@app.route('/submit', methods=['POST'])
def submit():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
    selected_model = request.form.get('model')
    selected_model +=".h5"
    
    if 'image_file' in request.files:
        image_file = request.files['image_file']
        #if image_file:
            #image_filename = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            #image_file.save(image_filename)
            #return f"Selected model: {selected_model} {image_filename}"
    #else:
        #return False
    modelPath = os.path.join("models/our_models",selected_model)
    model = load_model(modelPath)
    cnn_layers = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
    if not isinstance(model.layers[-1], Conv2D):
        cnn_layers.append(model.layers[-1].output)
    cnn_model = Model(inputs = model.input, outputs = cnn_layers)
    layer_names = cnn_model.output_names[:-1]
    img = image_file.stream
    img = Image.open(img)
    img = np.array(img)
    img = cv2.resize(img, cnn_model.input_shape[1:3])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    all_cnn = cnn_model.predict(img)
    send = {"output": all_cnn[-1].tolist(),
            "name_layers": []}
    #------------------------------------------
    for i in range(len(all_cnn)-1):
        helpMe = []
        for j in range(all_cnn[i].shape[-1]):
            img = Image.fromarray(cv2.cvtColor(np.uint8(all_cnn[i][0][:,:,j]), cv2.COLOR_BGR2RGB))
            image_stream = io.BytesIO()
            img.save(image_stream, format='PNG')
            helpMe.append(base64.b64encode(image_stream.getvalue()).decode('utf-8'))
        send["name_layers"].append({"name": layer_names[i], "feature_maps": helpMe, "num_feature_maps": len(helpMe)-1})
        send["modelPath"] = modelPath
    #------------------------------------------
    return render_template("result.html", data=send, num_layers=len(send["name_layers"]))

@app.route('/Visualization', methods=['POST'])
def visualize():
    uploaded_image = request.files['image2']   # Update the name to match your HTML input field name
    if uploaded_image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
        uploaded_image.save(image_path)
        #return f"Image uploaded successfully. Image saved at: {image_path}"
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.endswith(".h5"):
                model_name = file
                break
        uploaded_image.save(os.path.join("models/stolen models",model_name.split(".")[0], uploaded_image.filename))
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], file)

        model = load_model(model_path)
        cnn_layers = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
        if not isinstance(model.layers[-1], Conv2D):
            cnn_layers.append(model.layers[-1].output)
        cnn_model = Model(inputs = model.input, outputs = cnn_layers)
        layer_names = cnn_model.output_names[:-1]
        img = uploaded_image.stream
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img, cnn_model.input_shape[1:3])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        all_cnn = cnn_model.predict(img)
        send = {"output": all_cnn[-1].tolist(),
                "name_layers": []}
        #------------------------------------------
        for i in range(len(all_cnn)-1):
            helpMe = []
            for j in range(all_cnn[i].shape[-1]):
                img = Image.fromarray(cv2.cvtColor(np.uint8(all_cnn[i][0][:,:,j]), cv2.COLOR_BGR2RGB))
                image_stream = io.BytesIO()
                img.save(image_stream, format='PNG')
                helpMe.append(base64.b64encode(image_stream.getvalue()).decode('utf-8'))
            send["name_layers"].append({"name": layer_names[i], "feature_maps": helpMe, "num_feature_maps": len(helpMe)-1})
        #------------------------------------------
        print("heyyyy")
        print(len(send["name_layers"]))
        print(len(send["name_layers"][5]["feature_maps"]))
        print(send["name_layers"][5]["name"])
        print(send["name_layers"][5]["num_feature_maps"])
        #------------------------------------------
        send["modelPath"] = model_path
        return render_template("result.html", data=send, num_layers=len(send["name_layers"]))
    else:
        return "No image selected for upload."

@app.route('/Visualization2', methods=['POST'])
def visualize2():
    uploaded_image = request.files['image2']   # Update the name to match your HTML input field name
    if uploaded_image:
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) != 0:
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                if file.endswith(".h5"):
                    model_name = file
                    break
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            uploaded_image.save(image_path)
            print(model_name)
            print(os.path.join("models/stolen models",model_name.split(".")[0], uploaded_image.filename))
            uploaded_image.save(os.path.join("models/stolen models",model_name.split(".")[0], uploaded_image.filename))
                

        #return f"Image uploaded successfully. Image saved at: {image_path}"
        model_path = request.form["modelPath"]
        model = load_model(model_path)
        cnn_layers = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
        if not isinstance(model.layers[-1], Conv2D):
            cnn_layers.append(model.layers[-1].output)
        cnn_model = Model(inputs = model.input, outputs = cnn_layers)
        layer_names = cnn_model.output_names[:-1]
        img = uploaded_image.stream
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img, cnn_model.input_shape[1:3])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        all_cnn = cnn_model.predict(img)
        send = {"output": all_cnn[-1].tolist(),
                "name_layers": []}
        #------------------------------------------
        for i in range(len(all_cnn)-1):
            helpMe = []
            for j in range(all_cnn[i].shape[-1]):
                img = Image.fromarray(cv2.cvtColor(np.uint8(all_cnn[i][0][:,:,j]), cv2.COLOR_BGR2RGB))
                image_stream = io.BytesIO()
                img.save(image_stream, format='PNG')
                helpMe.append(base64.b64encode(image_stream.getvalue()).decode('utf-8'))
            send["name_layers"].append({"name": layer_names[i], "feature_maps": helpMe, "num_feature_maps": len(helpMe)-1})
        #------------------------------------------
        print("heyyyy")
        print(len(send["name_layers"]))
        print(len(send["name_layers"][5]["feature_maps"]))
        print(send["name_layers"][5]["name"])
        print(send["name_layers"][5]["num_feature_maps"])
        #------------------------------------------
        send["modelPath"] = model_path
        return jsonify(data=send, num_layers=len(send["name_layers"]))
    else:
        return "No image selected for upload."






if __name__ == '__main__':
   app.run(debug = True)