from flask import Flask, jsonify, request
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
import os

app = Flask(__name__)

# --- Cargar el pipeline de diarización ---
HF_TOKEN = os.getenv("HF_TOKEN")  
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-community-1")
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token=HF_TOKEN)


@app.route('/')
def home():
    return "Servidor de diarización de hablantes activo 🚀"

@app.route("/diarize", methods=["POST"])
def diarize():
    try:
        audio_file = request.files["audio"]
        temp_path = "temp.wav"
        audio_file.save(temp_path)

        # Ejecutar diarización
        diarization = pipeline(temp_path)

        # Convertir resultados a formato JSON
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })

        os.remove(temp_path)
        return jsonify({"segments": segments})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
