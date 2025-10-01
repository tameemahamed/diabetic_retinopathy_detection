import base64
from io import BytesIO
from flask import Flask, request, render_template, redirect, url_for
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

def pil_image_to_data_url(pil_img, fmt='PNG'):
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b = buf.getvalue()
    data_b64 = base64.b64encode(b).decode('ascii')
    return f"data:image/{fmt.lower()};base64,{data_b64}"

def preprocess_image_file(file_storage):
    """
    Accepts a Werkzeug FileStorage (request.files['original']) and processes in-memory.
    Returns:
      - original_input_batch (np.array, shape (1,640,640,3), normalized [-1,1])
      - bright_map_input (np.array, shape (1,640,640,3), normalized [-1,1])
      - dark_map_input (np.array, shape (1,640,640,3), normalized [-1,1])
      - bright_map_data_url (str)  <-- base64 data URL for embedding in template
      - dark_map_data_url (str)
    If processing fails, returns (None, None, None, None, None)
    """
    try:
        bytes_io = BytesIO(file_storage.read())
        img_pil = Image.open(bytes_io).convert("RGB")
        img_array = np.array(img_pil, dtype=np.uint8)
        cropped = crop_dark_border(img_array)
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return (None, None, None, None, None)

        resized = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)
        original_img_norm = (resized.astype(np.float32) / 127.5) - 1.0

        gray_resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        _, retina_mask = cv2.threshold(gray_resized, 20, 255, cv2.THRESH_BINARY)
        background_mask = cv2.bitwise_not(retina_mask)
        background_img_array = cv2.bitwise_and(resized, resized, mask=background_mask)
        background_img_norm = (background_img_array.astype(np.float32) / 127.5) - 1.0

        bright_map, dark_map = decompose_lesions(original_img_norm)

        # Convert maps to displayable PIL images (uint8 grayscale)
        bright_map_display = ((bright_map + 1.0) * 127.5).astype(np.uint8)
        dark_map_display = ((dark_map + 1.0) * 127.5).astype(np.uint8)

        # Ensure 2D -> convert to 3-channel for consistency with model input
        if bright_map_display.ndim == 2:
            bright_map_3ch = np.stack([bright_map_display, bright_map_display, bright_map_display], axis=-1)
        else:
            bright_map_3ch = bright_map_display

        if dark_map_display.ndim == 2:
            dark_map_3ch = np.stack([dark_map_display, dark_map_display, dark_map_display], axis=-1)
        else:
            dark_map_3ch = dark_map_display

        # Normalize to [-1, 1] for model
        bright_map_norm_3ch = (bright_map_3ch.astype(np.float32) / 127.5) - 1.0
        dark_map_norm_3ch = (dark_map_3ch.astype(np.float32) / 127.5) - 1.0
        background_img_3ch = np.stack([background_img_norm, background_img_norm, background_img_norm], axis=-1)

        # Create PIL images for embedding (grayscale)
        bright_pil = Image.fromarray(bright_map_display)
        dark_pil = Image.fromarray(dark_map_display)

        bright_data_url = pil_image_to_data_url(bright_pil, fmt='PNG')
        dark_data_url = pil_image_to_data_url(dark_pil, fmt='PNG')

        # Add batch dims
        original_input_batch = np.expand_dims(original_img_norm, axis=0)
        bright_input_batch = np.expand_dims(bright_map_norm_3ch, axis=0)
        dark_input_batch = np.expand_dims(dark_map_norm_3ch, axis=0)
        background_input_batch = np.expand_dims(background_img_3ch, axis=0)

        return original_input_batch, bright_input_batch, dark_input_batch, background_input_batch, bright_data_url, dark_data_url
    except Exception as e:
        print("Preprocessing error:", e)
        return (None, None, None, None, None)

# --- Flask Setup ---
app = Flask(__name__)
MODEL_PATH = 'dr_detection_model.h5'

# Load model (keep custom_objects)
try:
    model = load_model(MODEL_PATH, custom_objects={'SelfAttention': SelfAttention}, compile=False)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}. Ensure '{MODEL_PATH}' exists and is compatible.")
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
    if model is None:
        return render_template('index.html', error="Model not loaded. Check server logs.", result=False)

    if request.method == 'POST':
        # Validate if a file was uploaded
        if 'original' not in request.files or request.files['original'].filename == '':
            return render_template('index.html', error="Please upload an image file.", result=False)

        file = request.files['original']

        # Process entirely in memory; do NOT save to disk
        original_img, bright_map_input, dark_map_input, background_img_input, bright_data_url, dark_data_url = \
            preprocess_image_file(file)

        if original_img is None:
            return render_template('index.html', error="Failed to preprocess image. Ensure it's a valid retina image.", result=False)

        try:
            preds = model.predict([original_img, bright_map_input, dark_map_input], verbose=0)[0]
            grade = int(np.argmax(preds))
            confidence = float(preds[grade])
            class_name = CLASS_NAMES.get(grade, f'Unknown Class {grade}')
            confidence_scores = [float(p) for p in preds]

            # Return data URLs for embedded images instead of filesystem URLs
            return render_template(
                'index.html',
                result=True,
                grade=class_name,
                confidence=f"{confidence:.3f}",
                bright_map_url=bright_data_url,
                dark_map_url=dark_data_url,
                confidence_scores=confidence_scores,
                class_names=list(CLASS_NAMES.values())
            )
        except Exception as e:
            return render_template('index.html', error=f"Prediction failed: {e}", result=False)

    return render_template('index.html', result=False)

if __name__ == "__main__":
    # Local debug: use PORT env var if present
    port = int(os.environ.get("PORT", 5000))
    # NOTE: debug should be False for production (Render / Gunicorn)
    app.run(host="0.0.0.0", port=port, debug=False)
