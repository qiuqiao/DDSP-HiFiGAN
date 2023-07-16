from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from enhancer import Enhancer
import librosa
from f0_extractor import F0_Extractor
import soundfile as sf
import numpy as np
h = None
device = None

def enhance(a):
    # load input
    audio, sample_rate = librosa.load(a.input_wav_dir, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = h.hop_size * sample_rate / h.sampling_rate

    # extract f0
    f0_extractor=F0_Extractor('crepe',
                              sample_rate, 
                              hop_size, 
                              f0_min=40,
                              f0_max=1600
                              )
    f0=f0_extractor.extract(audio)
    audio_lenth=len(f0)*h.hop_size
    audio=np.concatenate([audio, np.zeros(audio_lenth-len(audio))])
    f0=torch.Tensor(f0).unsqueeze(0).unsqueeze(-1).to(device)
    enhancer = Enhancer(a.checkpoint_file, device=device)
    output, output_sample_rate = enhancer.enhance(
                                                torch.Tensor(audio).unsqueeze(0).to(device), 
                                                h.sampling_rate, 
                                                f0, 
                                                h.hop_size, 
                                                adaptive_key = a.adaptive_key)
    output = output.squeeze().cpu().numpy()
    sf.write(a.output_dir, output, output_sample_rate)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav_dir', default='i.wav')
    parser.add_argument('--output_dir', default='o.wav')
    parser.add_argument('--adaptive_key', default=0)
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    enhance(a)

if __name__ == '__main__':
    main()
