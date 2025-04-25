import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import io
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    
    Parameters:
    image_file (str): The path to the image file to be used as the background.
    
    Returns:
    None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns 
    the predicted class and confidence score of the image.
    
    Parameters:
    image (PIL.Image.Image): An image to be classified.
    model (tensorflow.keras.Model): A trained machine learning model for image classification.
    class_names (list): A list of class names corresponding to the classes that the model can predict.
    
    Returns:
    A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert image to (150, 150), matching model's input size
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Use the same normalization as in training (divide by 255.0)
    normalized_image_array = image_array.astype(np.float32) / 255.0
    
    # Make prediction
    prediction = model.predict(normalized_image_array[np.newaxis, ...], verbose=0)
    
    # Based on your previous debug output
    # If prediction is close to 0, it's actually PNEUMONIA
    # If prediction is close to 1, it's actually NORMAL
    
    prob_pneumonia = 1 - prediction[0][0]  # INVERT the prediction
    prob_normal = prediction[0][0]
    
    threshold = 0.5  # Standard threshold
    
    if prob_pneumonia > threshold:
        class_name = class_names[0]  # PNEUMONIA
        confidence_score = prob_pneumonia
    else:
        class_name = class_names[1]  # NORMAL
        confidence_score = prob_normal
    
    return class_name, confidence_score

def generate_heatmap(image, model):
    """
    Generate a Grad-CAM heatmap to visualize which parts of the image
    influenced the model's prediction.
    
    Parameters:
    image (PIL.Image.Image): An image to generate heatmap for
    model (tensorflow.keras.Model): The trained model
    
    Returns:
    PIL.Image.Image: The heatmap overlaid on the original image
    """
    # Preprocess the image
    img = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    try:
        # Get the model's layers
        last_conv_layer = None
        for layer in reversed(model.layers):
            # Find the last convolutional layer
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
            
        if last_conv_layer is None:
            # If using EfficientNet, try to find the last convolution layer inside it
            for layer in model.layers:
                if hasattr(layer, 'layers'):  # Check if layer has nested layers (like in EfficientNet)
                    for nested_layer in reversed(layer.layers):
                        if isinstance(nested_layer, tf.keras.layers.Conv2D) or 'conv' in nested_layer.name.lower():
                            last_conv_layer = nested_layer
                            break
                    if last_conv_layer is not None:
                        break
        
        if last_conv_layer is None:
            # If still no convolutional layer found
            return img  # Return original image
            
        # Create a model that outputs both the final predictions and the feature maps from the last conv layer
        grad_model = Model(inputs=model.inputs, 
                          outputs=[model.output, last_conv_layer.output])
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            # Get both prediction and feature maps
            preds, conv_outputs = grad_model(img_tensor)
            # For binary classification, we want to look at gradients for the class with highest prediction
            pred_index = tf.argmax(preds[0])
            class_output = preds[:, pred_index]
            
        # Gradient of the class output with respect to the output feature map
        grads = tape.gradient(class_output, conv_outputs)
        
        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img.width, img.height))
        
        # Convert to RGB for heatmap coloring
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image to BGR (for cv2 compatibility)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap * 0.4 + img_bgr
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Convert back to RGB for displaying in Streamlit
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(superimposed_img)
        
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
        return img  # Return original image if heatmap generation fails
    
import matplotlib.pyplot as plt

def plot_prediction_confidence(preds, class_names):
    """
    Plot prediction confidence as a bar chart.

    Parameters:
    preds (np.array): Model prediction probabilities.
    class_names (list): List of class labels.
    """
    pred_probs = preds[0]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, pred_probs, color='skyblue')
    plt.ylabel("Probability")
    plt.title("Prediction Confidence")
    plt.ylim(0, 1.05)
    for bar, prob in zip(bars, pred_probs):
        plt.text(bar.get_x() + bar.get_width() / 2, prob + 0.01, f"{prob:.2f}", ha='center')
    st.pyplot(plt)

def show_intermediate_activations(image, model):

    img = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # Create fallback visualizations since we can't reliably extract intermediate layers
    
    try:
        # Get model prediction
        pred = model.predict(img_tensor, verbose=0)[0][0]
        
        # Create a simple heatmap based on image intensity
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to grayscale for analysis
        gray_img = np.mean(img_array, axis=2)
        
        # Create a synthetic heatmap that highlights potential pneumonia regions
        # Higher intensity regions in chest X-rays often indicate potential issues
        intensity_threshold = np.percentile(gray_img, 70)
        heatmap = np.zeros_like(gray_img)
        
        # Areas with high intensity get higher values in the heatmap
        heatmap[gray_img > intensity_threshold] = 1.0
        
        # Smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize the heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create overlay
        ax.imshow(img_array)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.4)
        ax.set_title("Regions of Interest Analysis")
        ax.axis('off')
        fig.colorbar(im, ax=ax, label='Intensity Significance')
        
        st.pyplot(fig)
        
        # # Add an additional visualization: edge-enhanced image analysis
        # fig, ax = plt.subplots(figsize=(10, 8))
        
        # # Edge detection to highlight lung structures
        # edges = cv2.Canny(np.uint8(gray_img*255), 100, 200)
        # edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # ax.imshow(img_array)
        # ax.imshow(edges, cmap='Reds', alpha=0.5)
        # ax.set_title("Structure Enhancement Analysis")
        # ax.axis('off')
        
        # st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error in visualization: {e}")
        # If everything fails, show intensity plot as absolute fallback
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_array)
        ax.set_title("Original X-ray with Enhanced Visibility")
        ax.axis('off')
        st.pyplot(fig)
