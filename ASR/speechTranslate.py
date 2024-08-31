import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, AutoProcessor


def split_audio(audio, segment_duration=30, sampling_rate=16000):
    segment_samples = segment_duration * sampling_rate
    num_segments = int(np.ceil(len(audio) / segment_samples))
    segments = [audio[i*segment_samples:(i+1)*segment_samples] for i in range(num_segments)]
    return segments

def process_audio_file(file_path, segment_duration=30):
    audio, rate = librosa.load(file_path, sr=16000)
    segments = split_audio(audio, segment_duration=segment_duration)
    return segments

def load_and_preprocess_audio_segments(segments):
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    mel_spectrograms = [processor(segment, return_tensors="pt").input_features for segment in segments]
    return torch.cat(mel_spectrograms, dim=0)

def batch_inference(audio_files, batch_size=1, segment_duration=30):
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transcriptions = []

    for file_path in audio_files:
        segments = process_audio_file(file_path, segment_duration=segment_duration)
        inputs = load_and_preprocess_audio_segments(segments)
        inputs = inputs.to(device)

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            with torch.no_grad():
                generated_ids = model.generate(input_features=batch_inputs)
            batch_transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions.extend(batch_transcriptions)

    return transcriptions

# Exemple d'utilisation
audio_files = [ "/content/drive/MyDrive/train/alexa_lea_audio_0_train.wav",
    "/content/drive/MyDrive/train/alexa_lea_audio_10_train.wav",
    "/content/drive/MyDrive/train/alexa_lea_audio_12_train.wav"]
batch_size = 2
segment_duration = 30
transcriptions = batch_inference(audio_files, batch_size=batch_size)

for i, transcription in enumerate(transcriptions):
    print(f"Transcription for audio file {i}: {transcription}")
