import  tensorflow as tf
import  numpy as np

def prediction_with_model(model,image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels = 3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image,[24,24])
    image = tf.expand_dims(image,axis=0)

    prediction = model.predict(image)
    print(prediction)

    prediction = np.argmax(prediction)
    return prediction

if __name__ == "__main__":


    image_path8 = "C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\test\\0\\1.jpg"
    
    image_path = image_path8
    
    model = tf.keras.models.load_model("./model")
    prediction = prediction_with_model(model,image_path)
    
    print(prediction)

