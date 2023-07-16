import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from .nvSTFT import STFT
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    data,sampling_rate = librosa.load(full_path,sr=44100)
    # sampling_rate, data = read(full_path)
    # print(sampling_rate)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def get_dataset_filelist(a):
    training_files = [i for i in os.listdir(os.path.join(a.input_training_dir,'audio')) if i.endswith('.wav')]
    validation_files = [i for i in os.listdir(os.path.join(a.input_validation_dir,'audio')) if i.endswith('.wav')]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_audio_path=None,base_mels_path=None,base_f0_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.cached_f0 = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_audio_path = base_audio_path
        self.base_mels_path = base_mels_path
        self.base_f0_path = base_f0_path
        self.stft = STFT(
                sampling_rate, 
                num_mels, 
                n_fft, 
                win_size, 
                hop_size, 
                fmin, 
                fmax)

    def __getitem__(self, index):
        filename = self.audio_files[index].split('.')[0]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(os.path.join(self.base_audio_path,filename+'.wav'))
            # audio = audio / MAX_WAV_VALUE
            audio = audio
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.sampling_rate)
            self._cache_ref_count = self.n_cache_reuse
            f0=torch.from_numpy(np.load(os.path.join(self.base_f0_path,filename+'.npy')))
            f0 = f0.to(torch.float32)
            self.cached_f0 = f0
        else:
            audio = self.cached_wav
            f0 = self.cached_f0
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = self.stft.get_mel(audio)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, filename + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.shape[1] >= self.segment_size:
                    mel_start = random.randint(0, mel.shape[1] - frames_per_seg - 1)
                    mel = mel[:,:,mel_start:mel_start + frames_per_seg]
                    audio = audio[:,mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                    f0 = f0[mel_start:mel_start + frames_per_seg]
                # print((mel.shape, audio.shape, f0.shape))
                if (frames_per_seg - mel.shape[2]):
                    mel = torch.nn.functional.pad(mel, (frames_per_seg - mel.shape[2],0, 0,0, 0,0), 'constant')
                if self.segment_size - audio.shape[1]:
                    audio = torch.nn.functional.pad(audio, (self.segment_size - audio.shape[1],0, 0,0), 'constant')
                if frames_per_seg - f0.shape[0]:
                    f0 = torch.nn.functional.pad(f0, (frames_per_seg - f0.shape[0],0), 'constant')

        mel_loss = self.stft.get_mel(audio)
        # print((mel.shape, audio.shape, f0.shape, mel_loss.shape))
        return (mel.squeeze(), audio.squeeze(0), f0, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
