datapath = "/mnt/workspace/tdy.tdy/mp3_lww"
out_path = "/mnt/workspace/tdy.tdy/mp3_lww_train"
os.makedirs(out_path, exist_ok=True)
whisper_path = "/mnt/workspace/.cache/modelscope/keepitsimple/faster-whisper-large-v3"
target_language = 'zh'
buffer=0.2
eval_percentage=0.15
speaker_name="lww"

import os
from os import path as osp
import torchaudio
from matplotlib import pyplot as plt
import torch
from faster_whisper import WhisperModel
import pandas
import gc

# Loading Whisper
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Loading Whisper Model!")
asr_model = WhisperModel(whisper_path, device=device, compute_type="float16", local_files_only=True)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

print("Reading audio files!")
audio_files = os.listdir(datapath)
audio_total_size = 0
metadata = {"audio_file": [], "text": [], "speaker_name": []}
for f in audio_files:
    if f.endswith('mp3'):
        audio_path = osp.join(datapath, f)
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)
        # plot_waveform(wav, sr)
        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the begining of the file
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
                else:
                    # get previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence midle between the previous sentence and the current one
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start)/2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word
            
            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) / 2, word.end + buffer)
                
                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
                # if the audio is too short ignore it (i.e < 0.33 seconds)
                if audio.size(-1) >= sr/3:
                    torchaudio.save(absoulte_path,
                        audio,
                        sr
                    )
                else:
                    continue
                
                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)



df = pandas.DataFrame(metadata)
df = df.sample(frac=1)
num_val_samples = int(len(df)*eval_percentage)

df_eval = df[:num_val_samples]
df_train = df[num_val_samples:]

df_train = df_train.sort_values('audio_file')
train_metadata_path = os.path.join(out_path, "metadata_train.csv")
df_train.to_csv(train_metadata_path, sep="|", index=False)

eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
df_eval = df_eval.sort_values('audio_file')
df_eval.to_csv(eval_metadata_path, sep="|", index=False)

# deallocate VRAM and RAM
del asr_model, df_train, df_eval, df, metadata
gc.collect()

print('audio total size: ', audio_total_size)
