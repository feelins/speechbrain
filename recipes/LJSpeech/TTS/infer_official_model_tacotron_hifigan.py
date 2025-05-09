import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source=r"../../../pretrained_models/tts-tacotron2-ljspeech")
hifi_gan = HIFIGAN.from_hparams(source=r"../../../pretrained_models/tts-hifigan-ljspeech")

# Running the TTS
mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb, I don't believe it.")

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)