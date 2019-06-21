from flask import Flask, render_template, request,redirect,url_for,send_file
from flask_mail import Mail, Message
import os

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

if("upfolders" not in os.listdir("./static/")):
    os.mkdir("static/upfolders")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join("static","upfolders")

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

app.config.update(mail_settings)
mail = Mail(app)

mail.Message
# def init():

def train_model(folder_path):
    print('innn')
    import tensorflow as tf
    from tensorflow import keras    
    from PIL import Image
    import numpy as np
    import gc

    gc.enable()
    base_dir = folder_path
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
    epochs = 10
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print("training started")
    model.fit_generator(train_generator,
                                steps_per_epoch = steps_per_epoch,
                                epochs=epochs,
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=validation_steps)

    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit_generator(train_generator,
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=epochs,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps)

    model_path = folder_path+'_model_without_fine_tune.h5'
    model.save(model_path)
    gc.collect()
    return model_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["GET"])
def upload():
    return render_template("upload.html")


@app.route("/test",methods=["POST","GET"])
def test():
    accept_ext = ['zip','rar','7zip']
    if('zipfile' in request.files and any(True for ext in accept_ext if ext in request.files['zipfile'].filename)):
        import zipfile
        file = request.files['zipfile']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filepath)
        zip = zipfile.ZipFile(filepath)
        zip.extractall(path=app.config['UPLOAD_FOLDER'])
        zip.close()
        os.remove(filepath)
        folder_name = os.path.splitext(filepath)[0]
        print(folder_name)
        model_path = train_model(folder_name)
        return send_file(model_path,as_attachment=True,attachment_filename='model.h5')
    else:
        return "wrong uploaded file"


if __name__ == "__main__":
    # init()
    app.run(host="0.0.0.0",debug=True)
    # train_model()
    with app.app_context():
        msg = Message(subject="Hello",
                      sender=app.config.get("MAIL_USERNAME"),
                      recipients=["jaypatel9670@gmail.com"], 
                      body="This is a test email I sent with Gmail and Python!")
        mail.send(msg)