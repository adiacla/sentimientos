from flask import Flask, render_template, request, jsonify
import base64
import numpy as np

# Import functions and variables from our new modules
import config # Though not directly used here, ensures it's loaded if needed elsewhere
import utils
import model_loader

app = Flask(__name__)

# --- Load Artifacts at Startup ---
tokenizer = model_loader.load_tokenizer()
label_encoder, EMOTION_LABELS = model_loader.load_label_encoder() # Get both return values
model = model_loader.load_keras_model()

# --- Routes ---

@app.route("/")
def home():
    """Renders the home page."""
    return render_template("home.html")

@app.route("/about")
def about():
    """Renders the about page (if you have one)."""
    # Ensure you have an about.html template or remove this route if unused
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image OR text submission and emotion prediction."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_data_url = data.get("image")
    input_text = data.get("text") # Get potential text input

    ocr_text = None # Initialize ocr_text

    # --- Input Handling ---
    if image_data_url:
        # --- Image Processing Path ---
        if not image_data_url.startswith("data:image/png;base64,"):
            return jsonify({"error": "Invalid image data URL format"}), 400

        # Decode base64 image
        try:
            base64_string = image_data_url.split(",")[1]
            image_bytes = base64.b64decode(base64_string)
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return jsonify({"error": "Failed to decode image"}), 500

        # Process image and perform OCR
        img = utils.preprocess_image(image_bytes)
        if img is None:
            # Return a specific message that can be displayed to the user
            return jsonify({"text": "(Error)", "prediction": "Fallo al procesar la imagen"}), 200

        ocr_text = utils.perform_ocr(img)
        if ocr_text is None:
            # OCR failed, but maybe image processing worked
             return jsonify({"text": "(Error OCR)", "prediction": "Fallo el reconocimiento de texto (OCR)"}), 200
        if not ocr_text:
            # OCR succeeded but found no text
            return jsonify({"text": "", "prediction": "No se detectó texto en la imagen"}), 200
        # If OCR succeeded, ocr_text now holds the text to predict on

    elif input_text is not None: # Check if text was provided directly (allow empty string)
        # --- Text Input Path ---
        ocr_text = input_text # Use the provided text directly
        if not ocr_text:
             return jsonify({"text": "", "prediction": "No se proporcionó texto"}), 200
    else:
        # Neither image nor text provided
        return jsonify({"error": "Request must contain either 'image' or 'text'"}), 400

    # --- Text Preprocessing and Prediction (Common for both paths) ---
    prediction_result = "Model/Tokenizer/Encoder not loaded or preprocessing failed"

    # Check if all necessary components are loaded
    if model and tokenizer and label_encoder and EMOTION_LABELS:
        # Use preprocess_text_for_model from utils, passing the loaded tokenizer
        # ocr_text contains the text either from OCR or direct input
        processed_text = utils.preprocess_text_for_model(ocr_text, tokenizer)

        if processed_text is not None and processed_text.size > 0:
            try:
                predictions = model.predict(processed_text)[0]
                probs = { label: float(p) for label,p in zip(EMOTION_LABELS, predictions) }
                idx = np.argmax(predictions)
                # Ensure label_encoder.inverse_transform receives a list/array
                pred_emotion = label_encoder.inverse_transform([idx])[0]

                prediction_result = {
                    "emotion": pred_emotion,
                    "confidence": float(predictions[idx]),
                    "all_probabilities": probs
                }
                print(f"Input Text: '{ocr_text}' -> Prediction: {prediction_result}")

            except Exception as e:
                print(f"Error during model prediction: {e}")
                # Provide a more user-friendly error message if possible
                prediction_result = f"Predicción fallida: {type(e).__name__}"
        else:
            # This might happen if cleaning removes all characters
            prediction_result = "El texto quedó vacío después de la limpieza o el preprocesamiento falló"
            print(f"Text preprocessing returned None or empty array for input: '{ocr_text}'")
    else:
        missing = []
        if not model: missing.append("Modelo")
        if not tokenizer: missing.append("Tokenizer")
        if not label_encoder: missing.append("Codificador de Etiquetas")
        if not EMOTION_LABELS: missing.append("Etiquetas de Emoción")
        prediction_result = f"No se puede predecir, faltan componentes: {', '.join(missing)}"

    # Return the original text (from OCR or input) and the prediction result/status
    return jsonify({"text": ocr_text, "prediction": prediction_result})

# Optional: Add main execution block if running directly
# if __name__ == '__main__':
#    # Set debug=False for production
#    # Consider using a production server like gunicorn or waitress
#    app.run(debug=True, host='0.0.0.0', port=5000)
