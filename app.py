"""
Aplicación web basada en Flask que utiliza el modelo 'gemini-1.5-flash' de Google Gemini para generar respuestas
a partir de entradas del usuario. La aplicación almacena el historial de conversaciones y utiliza Markdown
para formatear las respuestas generadas.

Requisitos:
- Flask y Flask-Session instalados.
- Paquete `google.generativeai` instalado.
- Configurar la variable de entorno `GEMINI_API_KEY` con la clave de acceso a la API de Gemini.

Características:
1. Muestra una página inicial donde el usuario puede interactuar con el modelo.
2. Permite gestionar un historial limitado de interacciones entre usuario y modelo.
3. Formatea las respuestas en Markdown antes de mostrarlas.
"""

import os

import google.generativeai as genai
import jsonfy
import markdown
import requests
from flask import Flask, request, render_template, session
from flask_session import Session
from flask_cors import CORS

# Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://consultas.miopiamagna.org/"}})

# Configuración de sesiones
app.secret_key = "clave_secreta_para_sesiones"  # Cambia esto en producción
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# Configuración del modelo Gemini
genai.configure(api_key=os.getenv("GEMINI_ACCESS_TOKEN"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Constante para limitar el historial de interacciones
MAX_HISTORY = 4


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Endpoint de la API para recibir la pregunta del usuario y devolver la respuesta de Gemini en JSON.
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