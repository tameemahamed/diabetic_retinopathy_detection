import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv2D, Activation
from tensorflow.keras import backend as K
import cv2
from skimage.morphology import disk, opening, closing, black_tophat
import time

# --- Custom SelfAttention Layer ---
class SelfAttention(Layer):
    """
    Custom Self-Attention layer for integrating global contextual information.
    """
    def __init__(self, channels, name=None, **kwargs):
        super(SelfAttention, self).__init__(name=name, **kwargs)
        self.channels = channels
        # Convolutional layers for query, key, and value transformations
        self.query_conv = Conv2D(channels // 8, (1, 1), padding='same', name='query_conv')
        self.key_conv = Conv2D(channels // 8, (1, 1), padding='same', name='key_conv')
        self.value_conv = Conv2D(channels, (1, 1), padding='same', name='value_conv')
        # Learnable scalar parameter for scaling the attention output
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zeros', trainable=True)

    def build(self, input_shape):
        # Store height and width from input shape for reshaping
        self.height, self.width = input_shape[1], input_shape[2]
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.shape(inputs)[0]

        # Generate query, key, and value features
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)

        # Reshape query and key for matrix multiplication
        # q_: (batch_size, H*W, channels/8)
        q_ = K.reshape(q, (batch_size, -1, self.channels // 8))
        # k_: (batch_size, channels/8, H*W) - Transposed
        k_ = K.permute_dimensions(K.reshape(k, (batch_size, -1, self.channels // 8)), (0, 2, 1))

        # Calculate energy (dot product of query and key)
        # energy: (batch_size, H*W, H*W)
        energy = K.batch_dot(q_, k_) / K.sqrt(tf.cast(self.channels // 8, tf.float32))
        # Apply softmax to get attention weights
        attention = Activation('softmax')(energy)

        # Reshape value for matrix multiplication
        # v_: (batch_size, H*W, channels)
        v_ = K.reshape(v, (batch_size, -1, self.channels))

        # Compute the attention-weighted sum of values
        # out: (batch_size, H*W, channels)
        out = K.batch_dot(attention, v_)
        # Reshape back to image dimensions
        # out: (batch_size, H, W, channels)
        out = K.reshape(out, (batch_size, self.height, self.width, self.channels))

        # Apply gamma scaling and add residual connection
        return self.gamma * out + inputs

# --- Image Processing Functions ---
def suppress_vessels(image, kernel_length):
    """
    Suppresses blood vessels in the image using Gabor filters.
    Args:
        image (np.ndarray): Input grayscale image (uint8).
        kernel_length (int): Size of the Gabor kernel.
    Returns:
        np.ndarray: Vessel response map (float32, normalized to [0, 1]).
    """
    if kernel_length % 2 == 0:
        kernel_length += 1 # Ensure kernel length is odd
    angles = np.arange(0, 180, 15) # Angles for Gabor filters
    vessel_response = np.zeros_like(image, dtype=np.float32)

    for angle in angles:
        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            (kernel_length, kernel_length), sigma=kernel_length / 4.0,
            theta=np.deg2rad(angle), lambd=kernel_length / 2.0,
            gamma=0.5, psi=0
        )
        kernel -= kernel.mean() # Normalize kernel to have zero mean
        # Apply filter and accumulate maximum response
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        vessel_response = np.maximum(vessel_response, filtered)

    # Normalize the vessel response to [0, 1]
    if vessel_response.max() > 0:
        vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
    return vessel_response

def decompose_lesions(image):
    """
    Decomposes the image into bright and dark lesion maps.
    Args:
        image (np.ndarray): Input image (float32, normalized to [-1, 1]).
    Returns:
        tuple: (bright_map_norm, dark_map_norm) - Normalized bright and dark lesion maps.
    """
    # Convert image to uint8 for OpenCV operations (scale from [-1, 1] to [0, 255])
    image_for_cv = ((image + 1.0) * 127.5).astype(np.uint8)
    green_channel = image_for_cv[:, :, 1] # Use green channel for processing

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 82))
    green_proc = clahe.apply(green_channel)

    # Suppress vessels
    vessel_map = suppress_vessels(green_proc, 15)
    vessel_scaled = vessel_map * 0.8 # Scale down vessel suppression effect

    # Create lesion input by reducing vessel influence
    lesion_input = np.clip(green_proc * (1 - vessel_scaled), 0, 255)

    # --- Bright Lesion Map (Exudates) ---
    se_bright = disk(7) # Structural element for bright lesions
    opened = opening(lesion_input, se_bright) # Morphological opening
    bright_map = np.maximum(0, lesion_input - opened).astype(np.float32) # Calculate bright map
    # Normalize bright map
    bright_map = bright_map / bright_map.max() if bright_map.max() > 0 else bright_map
    bright_map[bright_map < 0.015] = 0 # Thresholding to remove noise

    # --- Dark Lesion Map (Hemorrhages, Microaneurysms) ---
    dark_maps = []
    # Apply black tophat filter with various disk sizes
    for radius in [3, 5, 10, 15, 20]:
        se_dark = disk(radius)
        top_hat = black_tophat(lesion_input, se_dark).astype(np.float32)
        if top_hat.max() > 0:
            top_hat /= top_hat.max() # Normalize each tophat result
        dark_maps.append(top_hat)

    dark_map = np.maximum.reduce(dark_maps) # Combine all dark maps
    dark_map[dark_map < 0.005] = 0 # Thresholding to remove noise

    # Post-processing for dark map (closing to fill small gaps)
    closed = closing((dark_map > 0).astype(np.uint8) * 255, disk(5))
    dark_map = dark_map * (closed / 255.0)

    # Normalize maps to [-1, 1] for model input
    bright_map_norm = (bright_map * 2.0) - 1.0
    dark_map_norm = (dark_map * 2.0) - 1.0

    return bright_map_norm, dark_map_norm

def crop_dark_border(image_np, threshold=10):
    """
    Crop out dark borders from a retinal image using a circular mask around the retina.
    Args:
        image_np (np.ndarray): Input image as a NumPy array (H, W, C).
        threshold (int): Intensity threshold for dark border removal.
    Returns:
        np.ndarray: Cropped image as a NumPy array.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply GaussianBlur and threshold to find the retinal region
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Find contours and identify the largest one (assumed to be the retina)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # If no contours found, return the original image
        return image_np

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the original image based on the bounding box of the largest contour
    cropped = image_np[y:y+h, x:x+w]
    return cropped

def preprocess_image_for_prediction(image_path, upload_folder, original_filename):
    """
    Preprocesses a single image by cropping dark borders, resizing,
    generating bright/dark lesion maps, and extracting the background.
    Also saves the bright and dark maps to the upload folder.
    Args:
        image_path (str): Path to the input image file.
        upload_folder (str): Path to the directory where images should be saved.
        original_filename (str): The original filename of the uploaded image.
    Returns:
        tuple: (input_original_batch, input_bright_map_batch, input_dark_map_batch,
                input_background_batch, bright_map_filename, dark_map_filename)
               or (None, None, None, None, None, None) if preprocessing fails.
    """
    bright_map_filename = None
    dark_map_filename = None
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil, dtype=np.uint8) # Convert to uint8 for OpenCV operations

        # Step 1: Remove dark borders
        cropped_img_array = crop_dark_border(img_array)

        # Step 2: Resize to target size (640x640)
        # Ensure the image is not empty after cropping
        if cropped_img_array.shape[0] == 0 or cropped_img_array.shape[1] == 0:
            print(f"Warning: Image {image_path} became empty after cropping. Skipping.")
            return None, None, None, None, None, None

        resized_img_array = cv2.resize(cropped_img_array, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize to [-1, 1] as required by the model
        original_img_norm = (resized_img_array.astype(np.float32) / 127.5) - 1.0

        # --- Background Extraction ---
        # Create a grayscale version of the resized image for mask generation
        gray_resized = cv2.cvtColor(resized_img_array, cv2.COLOR_RGB2GRAY)
        # Use a threshold to create a mask of the retinal area (foreground)
        # A threshold of 20 is commonly used to distinguish the retina from the dark background.
        _, retina_mask = cv2.threshold(gray_resized, 20, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the background area
        background_mask = cv2.bitwise_not(retina_mask)

        # Create the background image by applying the background mask to the original resized image.
        # This effectively blacks out the retinal area, leaving only the background.
        background_img_array = cv2.bitwise_and(resized_img_array, resized_img_array, mask=background_mask)

        # Normalize the background image to [-1, 1] for consistency
        background_img_norm = (background_img_array.astype(np.float32) / 127.5) - 1.0

        # Decompose lesions (uses original_img_norm)
        bright_map, dark_map = decompose_lesions(original_img_norm)

        # --- Save Bright and Dark Maps ---
        # Convert normalized maps [-1, 1] back to [0, 255] for saving as image
        bright_map_display = ((bright_map + 1.0) * 127.5).astype(np.uint8)
        dark_map_display = ((dark_map + 1.0) * 127.5).astype(np.uint8)

        # Ensure single channel for saving as grayscale images
        if bright_map_display.ndim == 3:
            bright_map_display = bright_map_display[:, :, 0]
        if dark_map_display.ndim == 3:
            dark_map_display = dark_map_display[:, :, 0]

        bright_map_pil = Image.fromarray(bright_map_display)
        dark_map_pil = Image.fromarray(dark_map_display)

        base_name = os.path.splitext(original_filename)[0]
        bright_map_filename = f"{base_name}_bright.png"
        dark_map_filename = f"{base_name}_dark.png"

        bright_map_filepath = os.path.join(upload_folder, bright_map_filename)
        dark_map_filepath = os.path.join(upload_folder, dark_map_filename)

        bright_map_pil.save(bright_map_filepath)
        dark_map_pil.save(dark_map_filepath)
        print(f"Saved bright map to: {bright_map_filepath}")
        print(f"Saved dark map to: {dark_map_filepath}")
        # --- End Save Bright and Dark Maps ---


        # Stack single-channel maps to 3 channels for model input consistency
        bright_map_3ch = np.stack([bright_map, bright_map, bright_map], axis=-1)
        dark_map_3ch = np.stack([dark_map, dark_map, dark_map], axis=-1)
        # Stack background map to 3 channels (even if not directly used by current model)
        background_img_3ch = np.stack([background_img_norm, background_img_norm, background_img_norm], axis=-1)


        # Add batch dimension to all processed images
        input_original_batch = np.expand_dims(original_img_norm, axis=0)
        input_bright_map_batch = np.expand_dims(bright_map_3ch, axis=0)
        input_dark_map_batch = np.expand_dims(dark_map_3ch, axis=0)
        input_background_batch = np.expand_dims(background_img_3ch, axis=0) # New extracted background input

        # Return all processed inputs, including the background image and map filenames
        return input_original_batch, input_bright_map_batch, input_dark_map_batch, input_background_batch, bright_map_filename, dark_map_filename
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None, None, None, None, None, None # Return None for all outputs in case of error

# --- Flask Setup ---
app = Flask(__name__)
MODEL_PATH = 'dr_detection_model.h5'
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set UPLOAD_FOLDER for send_from_directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model with error handling
try:
    # It's crucial to pass the custom_objects when loading a model that uses custom layers.
    model = load_model(MODEL_PATH, custom_objects={'SelfAttention': SelfAttention}, compile=False)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}. Please ensure '{MODEL_PATH}' exists and is a valid Keras model.")
    model = None

# --- Class Names for DR Grades ---
CLASS_NAMES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

# --- Flask Routes ---
@app.route('/home')
def home():
    """
    Renders the home page.
    """
    return render_template('home.html')

@app.route('/dr_info')
def dr_info():
    """
    Renders the DR information page.
    """
    return render_template('dr_info.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles image upload, preprocessing, and DR grade prediction.
    Displays the result or an error message.
    """
    # Check if the model was loaded successfully
    if model is None:
        return render_template('index.html', error="Model not loaded. Please check server logs for details.", result=False)

    if request.method == 'POST':
        # --- CLEANUP existing files on every new upload ---
        for fn in os.listdir(UPLOAD_FOLDER):
            fp = os.path.join(UPLOAD_FOLDER, fn)
            try:
                if os.path.isfile(fp): os.remove(fp)
            except Exception:
                pass
        
        # Validate if a file was uploaded
        if 'original' not in request.files or request.files['original'].filename == '':
            return render_template('index.html', error="Please upload an image file.", result=False)

        file = request.files['original']
        # Save the uploaded file temporarily
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image for prediction, now also getting the background image and map filenames
        original_img, bright_map_input, dark_map_input, background_img_input, bright_map_filename, dark_map_filename = \
            preprocess_image_for_prediction(filepath, UPLOAD_FOLDER, file.filename)

        # Clean up the temporary original file after processing
        # try:
        #     os.remove(filepath)
        # except Exception as e:
        #     print(f"Error removing temporary original file '{filepath}': {e}")

        # Check if preprocessing was successful (any of the returned values being None indicates failure)
        if original_img is None:
            return render_template('index.html', error="Failed to preprocess image. Please ensure it's a valid image file.", result=False)

        try:
            # Make prediction using the loaded model
            # IMPORTANT: The current model expects only 3 inputs.
            # So, we pass only 'original_img', 'bright_map_input', and 'dark_map_input'.
            preds = model.predict([original_img, bright_map_input, dark_map_input], verbose=0)[0]
            grade = int(np.argmax(preds)) # Get the predicted class index
            confidence = float(preds[grade]) # Get the confidence score for the predicted class
            class_name = CLASS_NAMES.get(grade, f'Unknown Class {grade}') # Map index to class name

            # Convert predictions to a list for the template
            confidence_scores = [float(pred) for pred in preds]

            # Render the result on the HTML page, passing URLs for the generated maps and confidence scores
            return render_template(
                'index.html',
                result=True,
                grade=class_name,
                confidence=f"{confidence:.3f}", # Format confidence to 3 decimal places
                bright_map_url=url_for('uploaded_file', filename=bright_map_filename),
                dark_map_url=url_for('uploaded_file', filename=dark_map_filename),
                confidence_scores=confidence_scores,
                class_names=list(CLASS_NAMES.values())
            )
        except Exception as e:
            # Catch any errors during prediction
            return render_template('index.html', error=f"Prediction failed: {e}. Please check the image format or model compatibility.", result=False)

    # Render the initial upload form for GET requests
    return render_template('index.html', result=False)

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    """
    Route to serve files from the UPLOAD_FOLDER.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Run the Flask application in debug mode (for development)
    app.run(debug=True)
    