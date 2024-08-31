from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoProcessor, WhisperForConditionalGeneration
import numpy as np
import io
import uvicorn
import librosa

app = FastAPI()

class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

# Charger les modèles et définir le dispositif
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle BERT personnalisé
model = CustomBert()
model.load_state_dict(torch.load("./my_custom_bert3.pth", map_location=device))
model.to(device)
model.eval()

# Liste des classes selon les labels de votre dataset
classes = ['0', '1']

# Charger le modèle Whisper pour la transcription
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
whisper_model.to(device)

def split_audio(audio, segment_duration=30, sampling_rate=16000):
    segment_samples = segment_duration * sampling_rate
    num_segments = int(np.ceil(len(audio) / segment_samples))
    segments = [audio[i*segment_samples:(i+1)*segment_samples] for i in range(num_segments)]
    return segments

def process_audio_file(audio_bytes, segment_duration=30):
    # Charger l'audio à partir des octets
    audio, rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    segments = split_audio(audio, segment_duration=segment_duration)
    return segments

def load_and_preprocess_audio_segments(segments):
    mel_spectrograms = [processor(segment, return_tensors="pt").input_features for segment in segments]
    return torch.cat(mel_spectrograms, dim=0)

def batch_inference(audio_bytes, batch_size=1, segment_duration=30):
    transcriptions = []
    segments = process_audio_file(audio_bytes, segment_duration=segment_duration)
    inputs = load_and_preprocess_audio_segments(segments)
    inputs = inputs.to(device)

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        with torch.no_grad():
            generated_ids = whisper_model.generate(input_features=batch_inputs)
        batch_transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcriptions.extend(batch_transcriptions)

    return transcriptions

def transcribe_audio(audio_bytes: bytes) -> str:
    transcriptions = batch_inference(audio_bytes, 1, 30)
    return transcriptions[0] if transcriptions else ""

def classifier_fn(text: str):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=250,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    _, pred = output.max(1)
    
    # Associer la prédiction à une étiquette lisible
    sentiment = "positif" if classes[pred.item()] == '1' else "négatif"
    return sentiment

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    print(f"Received file type: {file.content_type}")  # Log le type de fichier reçu
    if file.content_type not in ["audio/wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid audio file format")
    
    try:
        audio_bytes = await file.read()
        transcription = transcribe_audio(audio_bytes)
    except Exception as e:
        print(f"Error in transcription: {e}")
        raise HTTPException(status_code=500, detail="Error during transcription")

    try:
        prediction = classifier_fn(transcription)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Error during sentiment analysis")
    
    return {
        "transcription": transcription,
        "sentiment": prediction
    }

if __name__ == "__main__":
    import nest_asyncio 
    nest_asyncio.apply() # This allows nested event loops
    uvicorn.run(app, host="127.0.0.1", port=8989)
