import torch
import numpy as np
import random
import glob,os
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa


def create_qut_set(list_file_path,target_SNR_dB,speech_command_path,mixture_dir,noise):
    np.random.seed(10)
    spectrograms = []
    targets = []
    len_noise=len(noise)
    with open(list_file_path) as f:
        lines = f.readlines()
        for speech_path in lines:
            speech,sample_rate = torchaudio.load(os.path.join(speech_command_path,speech_path).strip())
            start = np.random.randint(len(noise)-32000)
            noise_cut = noise[start:start+16000]
            target = speech_path.split('/')[0]
            noise_cut = noise_cut.resize_as_(speech)
            scale = (torch.mean(speech.abs()**2)/(torch.mean(noise_cut.abs()**2) * 10.0**(target_SNR_dB/10.0)))**0.5
            mixture = speech + scale * noise_cut
            if not os.path.isdir(os.path.join(mixture_dir,target)):
                os.makedirs(os.path.join(mixture_dir,target))
            mixture_file_path = os.path.join(mixture_dir,speech_path).strip()
            torchaudio.save(mixture_file_path,mixture, sample_rate)



noise_path ='./data/QUT-NOISE/QUT-NOISE/QUT-NOISE/HOME-LIVINGB-1.wav'
noise, sr =librosa.load(noise_path,sr=16000)
noise = torch.tensor(noise)
speech_command_path= './data/SpeechCommands/speech_commands_v0.02'
def SNR_cal(speech_stft_mag,noise_stft_mag):
    return 10 * torch.log10(torch.mean(speech_stft_mag **2)/torch.mean(noise_stft_mag **2))

for target_SNR_dB in [-12.5,-10,0,10,20,30,40]:
    list_file_path = os.path.join(speech_command_path,'testing_list.txt')
    mixture_dir ='./data/SpeechCommands_QUT/speech_command_test_QUT_' + str(target_SNR_dB) +'_dB'
    create_qut_set(list_file_path,target_SNR_dB,speech_command_path,mixture_dir,noise)