import torch
import numpy as np
import random
import glob,os
import torchaudio

def whitenoise(length):
    noise_time = torch.normal(0.0,1.0,length)
    return noise_time

def make_whitenoise(wave_shape,batch_size,stft_transform):
    out = [stft_transform(whitenoise(wave_shape)) for i in range(batch_size)]
    out = torch.stack(out)
    out = torch.view_as_complex(out)
    return out

def musan_noise_list(musan_path = None):
    musan_files = []
    for f in glob.glob(os.path.join(musan_path,"*","*.wav")):
        musan_files.append(f)
    random.shuffle(musan_files)
    return musan_files

def make_musannoise(wave_shape,batch_size,stft_transform,musan_files):
    out = [stft_transform(torchaudio.load(random.choice(musan_files))[0]) for i in range(batch_size)]
    out = torch.stack(out)
    out = torch.view_as_complex(out)
    return out

def make_noise(wave_shape,batch_size,stft_transform,args):
    if not stft_transform:
        stft_transform = torchaudio.transforms.Spectrogram(n_fft = args.n_fft,hop_length = args.hop_length,power = None)
    if args.noise_type =='white':
        return make_whitenoise(wave_shape,batch_size,stft_transform)
    if args.noise_type =='constant':
        return make_constantnoise(wave_shape,batch_size,stft_transform,args.constant_value)
    else:
        musan_files = musan_noise_list(args.musan_path)    
        return make_musannoise(wave_shape,batch_size,stft_transform,musan_files)

def SNR_cal(speech_stft_mag,noise_stft_mag):
    return 10 * torch.log10(torch.mean(speech_stft_mag **2)/torch.mean(noise_stft_mag **2))

def whitenoise(length):
    noise_time = torch.normal(0.0,1.0,length)
    return noise_time

def make_constantnoise(wave_shape,batch_size,stft_transform,value=0.0001):
    constant_tensor = value * torch.ones(wave_shape)
    out = [stft_transform(constant_tensor) for i in range(batch_size)]
    out = torch.stack(out)
    out = torch.view_as_complex(out)
    return out