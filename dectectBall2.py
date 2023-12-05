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
#plt.plot(samples[100000:200000])
abs_samples=list(np.abs(samples_list))
plt.show()
plt.plot(abs_samples[100000:200000])
min(abs_samples)
#%%
sound.frame_rate
original_sample_rate
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
new_short=short_sound.low_pass_filter(int(frequencey_with_max_magnitude))
# %%
play(new_short)
# %%
play(short_sound)
#%%
frequencey_with_max_magnitude*45000/sound.frame_rate
# %%
def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
slow_sound = speed_change(new_short, 0.1)
# play(slow_sound+20)
#%%
slow_sound+=5
# %%
plt.figure(figsize=(12,7))
slow_sound_list=list(slow_sound.get_array_of_samples())
plt.plot(slow_sound_list)
# %%
file_handle = slow_sound.export("slow_sound_output.wav", format="wav")
# %%
len(slow_sound_list)
# %%
from scipy.signal import argrelmax
import pandas as pd
slow_sound_list_df=pd.Series(slow_sound_list)
smooth1_slow_sound_list=slow_sound_list_df.rolling(7).mean()
plt.plot(smooth1_slow_sound_list)
plt.plot(slow_sound_list)
relative_maximas=argrelmax(np.array(smooth1_slow_sound_list))
len(relative_maximas[0])
#%%
squares=list(map(lambda x:pow(x,7),slow_sound_list))
# squares=[float(x) if x>0 else 0.0 for x in squares]
plt.plot(squares)
relative_maximas=argrelmax(np.array(squares))
len(relative_maximas[0])


# %%
