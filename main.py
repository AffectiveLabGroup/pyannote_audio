from flask import Flask, jsonify, request
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
import os
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import huggingface_hub
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from io import BytesIO
import soundfile as sf


app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

# Registrar el token globalmente
huggingface_hub.login(HF_TOKEN)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

encoder = VoiceEncoder()

def extraer_firma(file_path):
    wav = preprocess_wav(file_path)
    emb = encoder.embed_utterance(wav)
    return emb

voces_conocidas = {
    "Paula": np.mean([
        extraer_firma("voices/paula_ref.wav"),
        extraer_firma("voices/paula_ref2.wav"),
        extraer_firma("voices/paula_ref3.wav"),
        extraer_firma("voices/paula_ref4.wav"),
    ], axis=0),
    "Loreto": np.mean([
        extraer_firma("voices/loreto_ref.wav"),
        extraer_firma("voices/loreto_ref2.wav"),
        extraer_firma("voices/loreto_ref3.wav"),
        extraer_firma("voices/loreto_ref4.wav"),
    ], axis=0),
    "Liany": np.mean([
        extraer_firma("voices/liany_ref.wav"),
        extraer_firma("voices/liany_ref2.wav"),
        extraer_firma("voices/liany_ref3.wav"),
        extraer_firma("voices/liany_ref4.wav"),
    ], axis=0),
    "Juan Jesus": np.mean([
        extraer_firma("voices/juanje_ref.wav"),
        extraer_firma("voices/juanje_ref2.wav"),
        extraer_firma("voices/juanje_ref3.wav"),
        extraer_firma("voices/juanje_ref4.wav"),
    ], axis=0)
}


@app.route('/')
def home():
    return "Servidor de diarización de hablantes activo 🚀"

@app.route("/diarize", methods=["POST"])
def diarize():
    try:
        audio_file = request.files["audio"]
        temp_path = "temp.wav"
        audio_file.save(temp_path)

        diarization = pipeline(temp_path)

        segments = []
        for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            archivo_fragmento = guardar_segmento(temp_path, turn.start, turn.end, speaker, idx)
            nombre_real = reconocer_voz(archivo_fragmento)
            segments.append({
                "speaker": nombre_real,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "file": archivo_fragmento
            })

        os.remove(temp_path)
        return jsonify({"segments": segments})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

def guardar_segmento(audio_path, start, end, speaker, idx):
    audio = AudioSegment.from_wav(audio_path)
    fragmento = audio[int(start*1000):int(end*1000)]  # convertir a milisegundos
    nombre_archivo = f"{speaker}_{idx}.wav"
    fragmento.export(nombre_archivo, format="wav")
    return nombre_archivo

def reconocer_voz(segmento_wav):
    firma = extraer_firma(segmento_wav)
    mejor_match = None
    distancia_min = float("inf")
    for nombre, firma_conocida in voces_conocidas.items():
        dist = euclidean(firma, firma_conocida)
        if dist < distancia_min:
            distancia_min = dist
            mejor_match = nombre
    return mejor_match


import uuid

@app.route("/upload", methods=["POST"])
def upload_audio():
    try:
        audio_file = request.files["audio"]

        if not audio_file:
            return jsonify({"error": "No se envió archivo"}), 400

        # Generar nombre único
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.wav"

        save_path = os.path.join("uploads", filename)

        os.makedirs("uploads", exist_ok=True)
        audio_file.save(save_path)

        return jsonify({
            "message": "Archivo guardado correctamente",
            "file_id": file_id,
            "filename": filename
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/diarize_live", methods=["POST"])
def diarize_live():
    try:
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({"error": "No se envió archivo"}), 400

        # Leer audio en memoria
        audio_bytes = BytesIO(audio_file.read())

        # Convertir a numpy array
        audio_bytes.seek(0)
        wav, sr = sf.read(audio_bytes)

        wav = wav.astype("float32")

        # Validar duración mínima (2 segundos mínimo)
        if len(wav) < sr * 2:
            return jsonify({"error": "Audio demasiado corto"}), 400

        # === DIARIZACIÓN ===
        # Pyannote puede trabajar con diccionario en memoria
        diarization = pipeline({"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": sr})

        turn, _, speaker = next(diarization.itertracks(yield_label=True))

        # Extraer segmento detectado
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        fragmento = wav[start_sample:end_sample]

        # Validar segmento mínimo
        if len(fragmento) < sr:
            return jsonify({"error": "Segmento demasiado corto"}), 400

        # === RECONOCIMIENTO DE VOZ ===
        firma = extraer_firma(fragmento)

        mejor_match = None
        distancia_min = float("inf")

        for nombre, firma_conocida in voces_conocidas.items():
            dist = euclidean(firma, firma_conocida)
            if dist < distancia_min:
                distancia_min = dist
                mejor_match = nombre

        if mejor_match is None:
            mejor_match = "desconocido"

        respuesta = f"Hola {mejor_match}, ¿cómo estás?"

        return jsonify({
            "respuesta": respuesta,
            "speaker": mejor_match,
            "distancia": float(distancia_min)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
