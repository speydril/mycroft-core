
from pathlib import Path
from time import perf_counter as timer
import soundfile as sf
import numpy as np
import torch

from cloning.encoder import inference as encoder
from cloning.synthesizer.inference import Synthesizer
from cloning.vocoder import inference as vocoder

recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]
class MycroftClone:
    def __init__(self):
        self.synthesizer = None
        self.wav = None

    def save_audio_file(self, wav, sample_rate):
        fpath = "./cloning/wav_files/file.wav"
        sf.write(fpath, wav, sample_rate)

    def synthesize(self, text:str , path):
        # Update the synthesizer random seed
        fpath = Path(path)
        wav = Synthesizer.load_preprocess_wav(fpath)
        seed = 43
        if seed is not None:
            torch.manual_seed(seed)
        # Synthesize the spectrogram
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        texts = text.split("\n")
        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

       
        self.current_generated = ("mycroft", spec, breaks, None)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        seed = 43
        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                    % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
           
     
        wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        print("vocoding something")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
        self.wav = wav / np.abs(wav).max() * 0.97
    

    def init_encoder(self):
        model_fpath = Path("./cloning/saved_models/default/encoder.pt")
        encoder.load_model(model_fpath)
  

    def init_synthesizer(self):
        model_fpath = Path("./cloning/saved_models/default/synthesizer.pt")
        self.synthesizer = Synthesizer(model_fpath)

    def init_vocoder(self):
        model_fpath = Path("./cloning/saved_models/default/vocoder.pt")
        vocoder.load_model(model_fpath)

    
if __name__ == "__main__":
    pass