from flask import Flask, render_template, request,redirect,url_for,send_file, jsonify
from flask_mail import Mail, Message
import os
import logging

if("upfolders" not in os.listdir("./static/")):
    os.mkdir("static/upfolders")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join("static","upfolders")

global log
log= ""

def init_smtp():
    with open('./emailenv.env','r') as f:
        global email
        global pwd
        data = f.read()
        email,pwd = data.split("#")
        print(len(email))

    
    mail_settings = {
        "MAIL_SERVER": 'smtp.gmail.com',
        "MAIL_PORT": 465,
        "MAIL_USE_TLS": False,
        "MAIL_USE_SSL": True,
        "MAIL_USERNAME": email,
        "MAIL_PASSWORD": pwd
    }
    global mail
    app.config.update(mail_settings)
    mail = Mail(app)

def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

# def init():

@app.route('/trainProgress')
def trainProgress():
    global log
    if(log):
        return jsonify({"Progress":log})
    else:
        return jsonify({"Progress":""})

@app.route('/download_model/<filename>')
def download_model(filename):
    print(filename)
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],filename),as_attachment=True,attachment_filename='model.h5')

@app.route('/train_model',methods=['POST','GET'])
def train_model():
    print('innn')
    import tensorflow as tf
    from tensorflow import keras
    from PIL import Image
    import numpy as np
    import gc

    global log
    print(request.json['folder'])
    folder_name = request.json['folder']
    log = r"Initializing variables...."
    gc.enable()
    base_dir = os.path.join(app.config['UPLOAD_FOLDER'],folder_name)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    image_size = 160
    batch_size = 32

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255)

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_mode='sparse')

    validation_generator = validation_datagen.flow_from_directory(
                    validation_dir, 
                    target_size=(image_size, image_size),
                    batch_size=batch_size,
                    class_mode='sparse')
    IMG_SHAPE = (image_size, image_size, 3)
    base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',classes=2)
    base_model.trainable = False
    model = keras.Sequential([base_model,
                            keras.layers.GlobalAveragePooling2D(),
                            keras.layers.Dense(2,activation='sigmoid')])
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    epochs = 1
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print("training started")
    log = r"Training the Model."
    model.fit_generator(train_generator,
                                steps_per_epoch = steps_per_epoch,
                                epochs=epochs,
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=validation_steps)

    fine_tune_at = 100
    log =  r"Dense Layer's Trainig done:."

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    
    log = log + r'Fine Tuning the model, beginning to train.'
    hist = model.fit_generator(train_generator,
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=epochs,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps)

    
    log = r'Model is done training,Acc:{}, Loss:{}, Val_acc:{}, Val_loss:{}'.format(hist.history['acc'],hist.history['loss'],hist.history['val_acc'],hist.history['val_loss'])
    model_path = base_dir + '_model_without_fine_tune.h5'
    model.save(model_path)
    # gc.collect()
    print(model_path)
    print(folder_name + '_model_without_fine_tune.h5')
    return jsonify({"success":True,"modelLink": folder_name + '_model_without_fine_tune.h5'})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["GET"])
def upload():
    return render_template("upload.html")


@app.route("/uploadFile",methods=['POST'])
def uploadFile(): 
    import zipfile
    file = request.files['zipfile']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filepath)
    zip = zipfile.ZipFile(filepath)
    zip.extractall(path=app.config['UPLOAD_FOLDER'])
    zip.close()
    os.remove(filepath)
    folder_name = os.path.splitext(filepath)[0]
    return jsonify({"redirect":True,
                    "redirectLink":"./start/"+os.path.splitext(file.filename)[0]})


@app.route('/start/<folder_name>')
def startTrain(folder_name):
    # path = train_model(os.path.join(app.config['UPLOAD_FOLDER'],folder_name))
    return render_template('startTrain.html')
    

# def unzipAndTrain(file):
#     import zipfile
#     file = request.files['zipfile']
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
#     file.save(filepath)
#     zip = zipfile.ZipFile(filepath)
#     zip.extractall(path=app.config['UPLOAD_FOLDER'])
#     zip.close()
#     os.remove(filepath)
#     folder_name = os.path.splitext(filepath)[0]
#     print(folder_name)
#     global log
#     log = log + r'Folders extraction complete.'
#     model_path = train_model(folder_name)


# @app.route("/test",methods=["POST","GET"])
# def test():
#     accept_ext = ['zip','rar','7zip']
#     if('zipfile' in request.files and any(True for ext in accept_ext if ext in request.files['zipfile'].filename)):
#         file = request.files['zipfile']
#         global log
#         log = r"File uploaded.. extracting...."
#         unzipAndTrain(file)
#         return render_template('test.html')
#         # return send_file(model_path,as_attachment=True,attachment_filename='model.h5')
#     else:
#         return "wrong uploaded file"


def send_msg():
    with app.app_context():
        msg = Message(subject="Hello",
                      sender=app.config.get("MAIL_USERNAME"),
                      recipients=["jaypatel9670@gmail.com"], 
                      body="This is a test email I sent with Gmail and Python!")
        mail.send(msg)

if __name__ == "__main__":
    # init()
    app.run(host="0.0.0.0",debug=True)
    # train_model()
    