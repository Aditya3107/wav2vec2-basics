from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from datasets import load_dataset, Audio
import datasets
import torch
import warnings
warnings.filterwarnings("ignore")

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/cache_model_dir")
processor = AutoProcessor.from_pretrained("/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="/vol/tensusers4/aparikh/Generic_Wav2vec2_models/Spanish_wav2vec2/cache_model_dir")
# load first sample utterance of decode_audio.csv 
decode_datasets = load_dataset("csv",data_files={"eval":"decoding_audio.csv"})
decode_datasets["eval"] = decode_datasets["eval"].cast_column("audiofilename", Audio(sampling_rate=16000))
sample = decode_datasets["eval"][0]
# forward sample through model to get greedily predicted transcription ids
input_values = processor(sample["audiofilename"]["array"], sampling_rate=16_000, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits[0].cpu().numpy()
# retrieve word stamps (analogous commands for `output_char_offsets`)
outputs = processor.decode(logits, output_word_offsets=True,alpha=0.7)
# compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
word_offsets = [
    {
      "word": d["word"],
      "start_time": round(d["start_offset"] * time_offset, 2),
      "end_time": round(d["end_offset"] * time_offset, 2),
  }
    for d in outputs.word_offsets
]
print(word_offsets)
print(outputs.text)