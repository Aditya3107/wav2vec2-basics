# Decoding Wav2vec2 model with and without LM
# Provides the transcripts for different values of LM weights. 
# You can set the weights of LM in alpha_val list 
# Input is providing the CSV file, set header "audiofilename"
# Resultant CSV is hypothesis transcripts : Without LM, With LM and different values of alpha.
# Aditya Parikh (CLST, Radboud University)
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import datasets
from datasets import DatasetDict, load_dataset, Audio
import pandas as pd
import numpy as np
import torch, librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


test_csv = "Spanish_wav2vec2/results_playground/sample.csv"
df = pd.read_csv(test_csv)
#df = df.drop(np.r_[2:30,79:100,267:741])
#df = df.dropna(inplace=True)
#df.reset_index(drop=True, inplace=True)


alpha_val = [0.00,0.50,0.70,0.80]
raw_datasets = load_dataset("csv",data_files={"eval":test_csv})
raw_datasets["eval"] = raw_datasets["eval"].cast_column("audiofilename", Audio(sampling_rate=16000))
transcript_withoutlm = []
transcript_withlm = {}


# import model, feature extractor, tokenizer
model_lm = AutoModelForCTC.from_pretrained("Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="Spanish_wav2vec2/cache_model_dir")
processor_lm = AutoProcessor.from_pretrained("Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="Spanish_wav2vec2/cache_model_dir")

model_wolm = Wav2Vec2ForCTC.from_pretrained("Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="Spanish_wav2vec2/cache_model_dir")
processor_wolm = Wav2Vec2Processor.from_pretrained("Spanish_wav2vec2/wav2vec2-common_voice-es-xls-r-300m",cache_dir="Spanish_wav2vec2/cache_model_dir")


for i in df["audiofilename"]:
    audio, rate = librosa.load(i, sr = 16000)
    input_values = processor_wolm(audio, sampling_rate=16_000, return_tensors = "pt", padding="longest").input_values
    logits = model_wolm(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = processor_wolm.batch_decode(prediction)[0]
    transcript_withoutlm.append(transcription)

df['transcript_withoutlm'] = transcript_withoutlm



for i in tqdm(range(df.shape[0])):
    sample = raw_datasets["eval"][i]
    input_values = processor_lm(sample["audiofilename"]["array"],sampling_rate=16_000,return_tensors="pt").input_values
    # forward sample through model to get greedily predicted transcription ids
    with torch.no_grad():
        logits = model_lm(input_values).logits[0].cpu().numpy()
    for alpha in alpha_val:
        variable_name = "transcript_alpha_" + str(alpha)
        outputs = processor_lm.decode(logits, output_word_offsets=True,alpha=alpha)
        df.at[i, variable_name] = outputs.text
    

df.to_csv("Spanish_wav2vec2/results_playground/sample_output.csv",index=False)

