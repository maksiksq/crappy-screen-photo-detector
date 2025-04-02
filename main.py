import numpy as np
from tensorflow import keras

model = keras.models.load_model("screen_photo_detector.keras")


def classify_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    print(f" Prediction: {prediction[0][0]}")

    if prediction[0][0] < 0.5:
        print("❌ PLEASE, FOR GOD'S SAKE, JUST MAKE A SCREENSHOT.")
    else:
        print("✅ This is a proper screenshot.")


classify_image("testq3.jpg")
