from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
warnings.filterwarnings("ignore")

model = Wav2Vec2ForCTC.from_pretrained("/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/cache_model_dir")
processor = Wav2Vec2Processor.from_pretrained("/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/cache_model_dir")
# Reading taken audio clip
import librosa, torch
audio, rate = librosa.load("/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/results_playground/audiofiles/common_voice_es_19987205.mp3", sr = 16000)
# Taking an input value
input_values = processor(audio, sampling_rate=16_000, return_tensors = "pt", padding="longest").input_values
# Storing logits (non-normalized prediction values)
logits = model(input_values).logits
# Storing predicted ids
prediction = torch.argmax(logits, dim = -1)
# Passing the prediction to the tokenizer decode to get the transcription
transcription = processor.batch_decode(prediction)[0]
print(transcription)