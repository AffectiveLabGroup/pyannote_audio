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
        extraer_firma("/voices/paula_ref.wav"),
        extraer_firma("/voices/paula_ref2.wav"),
        extraer_firma("/voices/paula_ref3.wav"),
        extraer_firma("/voices/paula_ref4.wav"),
    ], axis=0),
    "Loreto": np.mean([
        extraer_firma("/voices/loreto_ref.wav"),
        extraer_firma("/voices/loreto_ref2.wav"),
        extraer_firma("/voices/loreto_ref3.wav"),
        extraer_firma("/voices/loreto_ref4.wav"),
    ], axis=0),
    "Liany": np.mean([
        extraer_firma("/voices/liany_ref.wav"),
        extraer_firma("/voices/liany_ref2.wav"),
        extraer_firma("/voices/liany_ref3.wav"),
        extraer_firma("/voices/liany_ref4.wav"),
    ], axis=0),
    "Juan Jesus": np.mean([
        extraer_firma("/voices/juanje_ref.wav"),
        extraer_firma("/voices/juanje_ref2.wav"),
        extraer_firma("/voices/juanje_ref3.wav"),
        extraer_firma("/voices/juanje_ref4.wav"),
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


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
