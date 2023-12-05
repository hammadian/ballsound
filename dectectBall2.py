#%%
from pydub import AudioSegment
# import audiosegment
from pydub.playback import play
FILE_NAME='./sound_of_ball_original.wav'
from pydub.utils import mediainfo
info = mediainfo(FILE_NAME)
original_sample_rate=int(info['sample_rate'])
# sound = AudioSegment.from_file('./sound_of_ball_original.wav')
sound = AudioSegment.from_file(FILE_NAME)
# play(sound)
samples = sound.get_array_of_samples()
samples_list=samples.tolist()
import matplotlib.pyplot as plt
# plt.plot(samples)
plt.plot(samples[100000:200000])
# %%
one_sample=samples[100000:180000]
short_sound = AudioSegment(one_sample.tobytes(), frame_rate=sound.frame_rate,sample_width=sound.sample_width,channels=1)
play(short_sound)
# %%
import numpy as np
import os
import librosa
rosa_sound, sr=librosa.load(FILE_NAME, sr=int(original_sample_rate))

# %%
ft=np.fft.fft(rosa_sound)
ft_mag=np.abs(ft)
# %%
f_ratio=0.2
freq_x=np.linspace(0, original_sample_rate, len(ft_mag))
num_freq_bins=int(len(freq_x)* f_ratio)
plt.plot(freq_x[:num_freq_bins], ft_mag[:num_freq_bins])
# %%
ft_mag_list=ft_mag.tolist()
frequencey_with_max_magnitude=freq_x[ft_mag_list.index(max(ft_mag_list))]
new = sound.low_pass_filter(int(frequencey_with_max_magnitude))

# %%
play(new)
# %%
play()