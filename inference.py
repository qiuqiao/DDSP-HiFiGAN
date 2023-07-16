import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
from pc_ddsp.ddsp.vocoder import load_model, Audio2Mel
from hifigan.enhancer import Enhancer
import json
from hifigan.env import AttrDict

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dm",
        "--ddsp_model_path",
        type=str,
        required=False,
        default='pretrained/ddsp/model_best.pt',
        help="path to the ddsp model file",
    )
    parser.add_argument(
        "-hm",
        "--hifigan_model_path",
        type=str,
        required=False,
        default='pretrained/hifigan/model',
        help="path to the nsf-hifigan model file",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones)",
    )
    parser.add_argument(
        "-ak",
        "--adaptive_key",
        type=str,
        required=False,
        default=0,
        help="adapt to a higher vocal range, mostly equals to -k (number of semitones)",
    )
    return parser.parse_args(args=args, namespace=namespace)
    
if __name__ == '__main__':
    
    # cpu inference is fast enough!
    device = 'cpu' 
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load model
    model, args = load_model(cmd.ddsp_model_path, device=device)
    
    sampling_rate = args.data.sampling_rate
    hop_length = args.data.block_size
    win_length = args.data.win_length
    n_fft = args.data.n_fft
    n_mel_channels = args.data.n_mels
    mel_fmin = args.data.mel_fmin
    mel_fmax = args.data.mel_fmax
    
    # load input
    x, _ = librosa.load(cmd.input, sr=sampling_rate)
    x_t = torch.from_numpy(x).float().to(device)
    x_t = x_t.unsqueeze(0).unsqueeze(0) # (T,) --> (1, 1, T)
    
    # mel analysis
    mel_extractor = Audio2Mel(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        n_mel_channels=n_mel_channels,
        win_length=win_length,
        n_fft=n_fft,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax).to(device)
    
    mel = mel_extractor(x_t)
     
    # f0 analysis using dio
    '''
    _f0, t = pw.dio(
           x.astype('double'), 
           sampling_rate, 
           f0_floor=65.0, 
           f0_ceil=1047.0, 
           channels_in_octave=2, 
           frame_period=(1000*hop_length / sampling_rate))
    f0 = pw.stonemask(x.astype('double'), _f0, t, sampling_rate)
    f0 = f0.astype('float')
    '''
    
    # f0 analysis using parselmouth (faster)
    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
            time_step=hop_length / sampling_rate, voicing_threshold=0.6,
            pitch_floor=65, pitch_ceiling=800).selected_array['frequency']
    pad_size=(int(len(x) // hop_length) - len(f0) + 1) // 2
    f0 = np.pad(f0,[[pad_size,mel.size(1) - len(f0) - pad_size]], mode='constant')
    
    # interpolate the unvoiced f0 
    uv = f0 == 0
    f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
   
    # key change
    f0 = f0 * 2**(float(cmd.key)/12)
     
    # forward and save the output
    with torch.no_grad():
        print(mel.shape,f0.shape)
        signal, _, (s_h, s_n) = model(mel, f0)
        signal = signal.squeeze().cpu().numpy()
    sf.write('ddsp_'+cmd.output,signal ,44100)
    
    #load hifigan config
    config_file = os.path.join(os.path.split(cmd.hifigan_model_path)[0], 'config.json')
    with open(config_file) as f:
        cfg = f.read()
    json_config = json.loads(cfg)
    h = AttrDict(json_config)

    enhancer = Enhancer(cmd.hifigan_model_path, device=device)
    output, output_sample_rate = enhancer.enhance(
                                                torch.Tensor(signal).unsqueeze(0).to(device), 
                                                h.sampling_rate, 
                                                f0, 
                                                h.hop_size, 
                                                adaptive_key = cmd.adaptive_key)
    output = output.squeeze().cpu().numpy()
    sf.write(cmd.output, output, output_sample_rate)
