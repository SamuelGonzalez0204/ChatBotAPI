import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import whisper
from gtts import gTTS
from flask import Flask, request, jsonify

# Cargar variables de entorno
load_dotenv()
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

# Modelos
model_name = "meta-llama/Llama-2-7b-chat-hf"
model_id = "openai/whisper-large-v3"

# Configurar Flask
app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Cargar modelo Whisper
whisper_model = whisper.load_model("medium")

# Cargar modelo Llama 2
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True, use_auth_token=access_token
)

# Función para obtener respuesta de Llama 2
def obtener_respuesta_llama2(pregunta):
    prompt = f"Responde con información detallada y clara. Usuario: {pregunta}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llama_model.generate(**inputs, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Función para transcribir audio con Whisper
def transcribir_audio(archivo_audio):
    resultado = whisper_model.transcribe(archivo_audio)
    return resultado["text"]

# Función para convertir texto a voz
def texto_a_voz(texto, nombre_archivo="respuesta.mp3"):
    tts = gTTS(texto, lang="es")
    tts.save(nombre_archivo)
    return nombre_archivo

# Ruta para el chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    file = request.files["audio"]
    file.save("temp.mp3")

    # Transcribir audio
    texto = transcribir_audio("temp.mp3")

    # Obtener respuesta de Llama 2
    respuesta = obtener_respuesta_llama2(texto)

    # Convertir a voz
    texto_a_voz(respuesta, "respuesta.mp3")

    return jsonify({"texto": respuesta, "audio": "respuesta.mp3"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
