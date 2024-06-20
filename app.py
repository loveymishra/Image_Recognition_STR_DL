import streamlit as st
from PIL import Image
import tensorflow as tf

class_names = ["Audi", "Creta", "Scorpio", "Rolls-Royce", "Swift",
                "Safari", "Innova"]

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224)) 
    img = tf.convert_to_tensor(img)  
    img = tf.cast(img, dtype=tf.float32)  
    img = img / 255.0  
    img = tf.expand_dims(img, axis=0)  
    return img


def make_prediction(model, image):
    predictions = model.predict(image)  
    sorted_indices = tf.argsort(predictions[0])[::-1]  
    predicted_class = class_names[sorted_indices[0]]
    second_highest_class = class_names[sorted_indices[1]]
    return predicted_class, second_highest_class, predictions[0], sorted_indices


def main():
    st.title("F1 Car Image Classification App")

    st.write("Upload an image to classify it. The app supports the following car brands:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - Audi
        - Creta
        - Scorpio
        - Rolls-Royce
        """)
    with col2:
        st.write("""
        - Swift
        - Safari
        - Innova
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing the image..."):
                image = preprocess_image(uploaded_file)

                model = tf.keras.models.load_model(r"C:\Users\PC\Desktop\DNN_Models\Car_image_classification_M4_94a.h5")  # Replace with your model path

                predicted_class, second_highest_class, probabilities, sorted_indices = make_prediction(model, image)


                st.subheader(
                    f"The car shown in the image is   : {predicted_class} ({probabilities[sorted_indices[0]] * 100:.2f}%)")
                st.write(
                    f"The second most probable car is   : {second_highest_class} ({probabilities[sorted_indices[1]] * 100:.2f}%)")
                st.image(uploaded_file)



        except Exception as e:
            st.error(f"Error processing image: {e}")


main()
