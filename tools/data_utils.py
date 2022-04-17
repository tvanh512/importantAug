import os  
import torch  
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


#from pytorch19:  https://github.com/pytorch/audio/blob/main/torchaudio/datasets/speechcommands.py
class SubsetSC(SPEECHCOMMANDS):
    
    def __init__(self, data_path,subset: str = None,url="speech_commands_v0.02"):
        super().__init__(root=data_path,url=url,download=True)
        self.data_path = data_path
        self.folder_in_archive = "SpeechCommands"
        self.HASH_DIVIDER = "_nohash_"
        self.EXCEPT_FOLDER = "_background_noise_"

        folder_in_archive = os.path.join(self.folder_in_archive, url)
        self.path = os.path.join(data_path,folder_in_archive)
        print('self.path',self.path)
        def _load_list(root, *filenames):
            output = []
            for filename in filenames:
                filepath = os.path.join(root, filename)
                with open(filepath) as fileobj:
                    output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
            return output

        if subset == "validation":
            self._walker = _load_list(self.path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self.path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self.path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self.path).glob('*/*.wav'))

            self._walker = [
                w for w in walker
                if self.HASH_DIVIDER in w
                and self.EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
            ]

def path_index(dataset):
    path_to_index = {}
    index_to_path = {}
    index = 0
    for waveform, sample_rate, label, speaker_id, utterance_number in dataset:
        path = os.path.join(label,speaker_id +'_nohash_' + str(utterance_number) +'.wav' )
        path_to_index[path] = index
        index_to_path[index] = path
        index+=1
    return path_to_index,index_to_path

# from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
def label_to_index(word,labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

# from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
def index_to_label(index,labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

# From Nvidia Nemo
def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2

def data_processing(data,data_type,labels,stft_transform,args,path_to_index=None):
    spectrograms = []
    targets = []
    indexes = []

    for (waveform, sample_rate, target_text, speaker_id, utterance_number) in data: # waveform, sample_rate, label, speaker_id, utterance_number
        if data_type == 'train':
            spec = stft_transform(waveform).squeeze(0).transpose(0, 1).contiguous()
        else:
            spec = stft_transform(waveform).squeeze(0).transpose(0, 1).contiguous()
            path = os.path.join(target_text,speaker_id +'_nohash_' + str(utterance_number) +'.wav')
            indexes += [torch.tensor(path_to_index[path])]
        spectrograms.append(spec)
        targets += [label_to_index(target_text,labels)]

    targets = torch.stack(targets)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3).contiguous()
    if data_type == 'train':
        return spectrograms, targets
    else:
        indexes = torch.stack(indexes)
        return spectrograms, targets,indexes

def make_relativepath_index(file_paths):
    path_to_index = {}
    index_to_path = {}
    index = 0
    with open(file_paths) as f:
        lines = f.readlines()
        for path in lines:
            path = path.strip()
            path_to_index[path] = index
            index_to_path[index] = path
            index+=1
    return path_to_index,index_to_path

def musan_loader(file_paths,musan_path,speech_command_path,batch_size,path_to_index,audio_transforms):
    spectrograms = []
    targets = []  
    indexes = []
    with open(file_paths) as f:
        lines = f.readlines()
        for mixture_path in lines:
            mixture,sample_rate = torchaudio.load(os.path.join(musan_path,mixture_path).strip())
            target = mixture_path.split('/')[0]
            targets += [label_to_index(target,labels)]
            spec = audio_transforms(mixture).squeeze(0).transpose(0, 1).contiguous()
            spectrograms.append(spec)
            indexes += [torch.tensor(path_to_index[mixture_path])]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3).contiguous()
    targets = torch.stack(targets)
    indexes = torch.stack(indexes)
    musan_dataset = TensorDataset(spectrograms,targets,indexes) # create your datset
    musan_dataloader = DataLoader(musan_dataset, batch_size=batch_size) 
    return musan_dataloader

def load_config(config_path,config_name):
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(config_path)
    config = config._sections[config_name]
    return config

class step_counter(object):
    def __init__(self):
        self.count = 0

    def increase_one(self):
        self.count += 1

    def get(self):
        return self.count 


class NoisyDataset(Dataset):

    def __init__(self, path_to_index, root_dir, transform):
        self.file_paths = list(path_to_index.keys())
        self.root_dir = root_dir
        self.transform = transform
        self.path_to_index = path_to_index
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx] #e.g right/837a0f64_nohash_0.wav
        target = file_path.split('/')[0]
        index = self.path_to_index[file_path]
        wav_full_path = os.path.join(self.root_dir,file_path)
        waveform,sample_rate = torchaudio.load(wav_full_path)
        # shape [F, T, 2] --> [T,F,2]
        spectrogram = self.transform(waveform).squeeze(0).transpose(0,1).contiguous() 
        #print(spectrogram.shape)
        return spectrogram, target,index
        
def data_processing_noisy(data,labels):
    spectrograms = []
    targets = []
    indexes = []

    for (spec, target,index) in data: 
        indexes += [torch.tensor(index)]
        spectrograms.append(spec)
        
        targets += [label_to_index(target,labels)]

    targets = torch.stack(targets)
    
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3).contiguous()
    return spectrograms, targets,indexes