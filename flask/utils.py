import re
import unicodedata
import cv2
import numpy as np
import easyocr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_SEQUENCE_LENGTH

# Initialize easyocr Reader (load the model only once)
# Specify the language(s) you want to detect, e.g., ['en'] for English
print("--- Initializing EasyOCR Reader ---")
# This might take a moment when the application starts
reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a compatible GPU and PyTorch installed
print("--- EasyOCR Reader Initialized ---")

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Image preprocessing for OCR
def preprocess_image(img_bytes):
    """Converts image bytes to OpenCV format and applies preprocessing for OCR."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: cv2.imdecode failed.")
            return None

        # 1. Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply adaptive thresholding
        # Use THRESH_BINARY_INV because input is black text on white background
        # Parameters (block size, C) might need tuning
        img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 8)

        # 3. Apply morphological opening to remove small noise/dots
        # Kernel size might need tuning
        kernel = np.ones((2,2), np.uint8)
        img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

        # Optional: Add slight blur if needed (before or after thresholding)
        # img_blur = cv2.medianBlur(img_gray, 3) # Example before thresholding
        # img_blur = cv2.medianBlur(img_clean, 3) # Example after opening

        # Optional: Deskewing could be added here if necessary

        # Return the preprocessed image (binary)
        return img_clean
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Perform OCR
def perform_ocr(img):
    """Performs OCR on the preprocessed image using EasyOCR."""
    if img is None:
        print("OCR skipped: Input image is None.")
        return None
    try:
        # Use easyocr reader
        # detail=0 returns only the text
        # paragraph=True attempts to join nearby text boxes into paragraphs
        results = reader.readtext(img, detail=0, paragraph=True)
        text = " ".join(results) # Join detected text blocks
        print(f"EasyOCR Result: '{text}'")
        return text.strip()
    except Exception as e:
        print(f"Error during EasyOCR: {e}")
        return None

# Preprocess text for the model
def preprocess_text_for_model(text, tokenizer):
    """Preprocesses text using loaded tokenizer and padding."""
    if not text or not tokenizer:
        print("Preprocessing failed: No text or tokenizer.")
        return None
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            print("Preprocessing failed: Text empty after cleaning.")
            return None

        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences(
            sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post" # Use imported constant
        )

        print(
            f"Original OCR: '{text}' -> Cleaned: '{cleaned_text}' -> Padded Shape: {padded_sequences.shape}"
        )
        return padded_sequences

    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return None

