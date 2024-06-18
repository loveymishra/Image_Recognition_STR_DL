import streamlit as st
from PIL import Image
import tensorflow as tf  # Assuming TensorFlow for your model

def load_and_preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, dtype=tf.float32)  # Convert to float32
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)
    return img

# Function to make predictions using your loaded model
def make_prediction(model, image):
    predictions = model.predict(image)  # Pass the preprocessed image to your model
    predicted_class = tf.argmax(predictions[0]).numpy()  # Get the index of the highest probability class
    class_names = ["Audi", "Creta", "scorpio", "Rolls Royce", "swift",
                "Safari", "Innova"]  # Replace with your actual class names
    return class_names[predicted_class]

def main():
    st.title("Image Classification App")
    st.write("Only Of These Brands : ['Audi' , 'Hyundai Creta' , 'Mahindra Scorpio' , 'Rolls Royce' , 'Swift' , 'Tata Safari' , 'Toyota Innova']")
    st.write("Prediction Accuracy is 87%")
    # Display instructions
    st.write("Upload an image.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Load and preprocess the image
        image = load_and_preprocess_image(uploaded_image)

        # Load your pre-trained model (replace with your model loading logic)
        model = tf.keras.models.load_model("Car_image_classification_M4_94a.h5")  # Replace with your model path

        # Show a spinner while making the prediction
        with st.spinner('Making prediction...'):
            # Make prediction and display results
            prediction = make_prediction(model, image)

        st.subheader("Prediction:")
        st.write("Car Shown In Image is : "+prediction)

        # Optionally display the uploaded image
        st.image(uploaded_image, caption=prediction, use_column_width=True)

main()
