import os
import google.generativeai as genai
import json
import markdown
import requests
from flask import Flask, request, session, jsonify
from flask_session import Session
from flask_cors import CORS
import whisper

# Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://consultas.miopiamagna.org/"}})
app.secret_key = "clave_secreta_para_sesiones"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Cargar modelo Whisper (asegúrate de que el tamaño sea apropiado para tus recursos)
whisper_model = whisper.load("base")

# Configuración del modelo Gemini
genai.configure(api_key=os.getenv("GEMINI_ACCESS_TOKEN"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Constante para limitar el historial de interacciones
MAX_HISTORY = 4

# Función para transcribir audio con Whisper
def transcribir_audio(archivo_audio):
    resultado = whisper_model.transcribe(archivo_audio)
    return resultado["text"]

@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """
    Endpoint para recibir un archivo de audio y transcribirlo a texto usando Whisper.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No se proporcionó ningún archivo de audio"}), 400

    audio_file = request.files['audio']
    try:
        # Guardar temporalmente el archivo de audio
        audio_file.save("temp_audio.mp3")
        transcribed_text = transcribir_audio("temp_audio.mp3")
        os.remove("temp_audio.mp3")  # Eliminar el archivo temporal
        return jsonify({"text": transcribed_text})
    except Exception as e:
        return jsonify({"error": f"Error al transcribir el audio: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Endpoint para recibir el texto del usuario y obtener la respuesta de Gemini.
    """
    prompt = request.form.get("prompt")
    if not prompt:
        return jsonify({"error": "Por favor, ingresa un texto válido."}), 400

    history = session.get("history", [])
    context = ""
    for item in history[-MAX_HISTORY:]:
        context += f"Usuario: {item['prompt']}\nModelo: {item['response_raw']}\n"
    context += f"Usuario: {prompt}\n"

    try:
        response = model.generate_content(context).text
        output_html = markdown.markdown(response)

        history.append({
            "prompt": prompt,
            "response_raw": response,
            "response_html": output_html
        })
        session["history"] = history

        return jsonify({"response_html": output_html})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error al conectarse a Gemini: {e}"}), 500

@app.route("/")
def home():
    return "Esta es la API del chatbot. El frontend está en WordPress."

if __name__ == "__main__":
    app.run(debug=True)
