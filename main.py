

from flask import Flask, jsonify, request
from pyannote.audio import Model
from scipy.spatial.distance import cosine 
import torch
import torchaudio
import os

app = Flask(__name__)


# --- Cargar modelo de embeddings ---
HF_TOKEN = os.getenv("HF_TOKEN")
#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return "Servidor de reconocimiento de speakers activo 🚀"

# --- Cargar embeddings conocidos ---
KNOWN_EMBEDDINGS = {}

def get_embedding(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    with torch.no_grad():
        emb = model({"waveform": waveform.to(device), "sample_rate": sample_rate})
    return emb.cpu().numpy().mean(axis=1)

# Cargar embeddings pregrabados
def load_known_embeddings():
    base_path = "./voices"
    for name in os.listdir(base_path):
        if name.endswith(".wav"):
            path = os.path.join(base_path, name)
            embedding = get_embedding(path)
            KNOWN_EMBEDDINGS[name.replace(".wav", "")] = embedding

load_known_embeddings()


@app.route("/recognize", methods=["POST"])
def recognize():
    audio_file = request.files["audio"]
    temp_path = "temp.wav"
    audio_file.save(temp_path)

    new_emb = get_embedding(temp_path)

    best_match = "Desconocido"
    best_score = 0.0

    for name, emb in KNOWN_EMBEDDINGS.items():
        sim = 1 - cosine(new_emb, emb)
        if sim > best_score:
            best_score = sim
            best_match = name

    # Umbral de confianza
    if best_score < 0.75:
        best_match = "Desconocido"

    print(f"Identificado: {best_match} (sim={best_score:.2f})")
    return jsonify({"speaker": best_match, "similarity": best_score})

# DIARIZACION DE HABLANTES
# huggingface-token: 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
#@app.route("/stream_audio", methods=["POST"])
#def stream_audio():
#    audio_bytes = request.
#    
    # Guardar temporalmente el audio recibido en un fichero temporal
#    with open("temp_audio.wav", "wb") as f:
#        f.write(audio_bytes)
    
#    try:
#        diarization = pipeline("temp_audio.wav")
#        speakers = {f"SPEAKER_{s.label}" for s in diarization.itertracks(yield_label=True)}
#        os.remove("temp_audio.wav")
#        return ",".join(speakers)

#    except Exception as e:
#        print("Error:", e)
#        return "error", 500

if __name__ == "__main__":
    from flask import Flask
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))