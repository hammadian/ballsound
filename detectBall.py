#%%
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statistics
import collections
import pandas as pd
from pydub.utils import mediainfo

#convert audio to datasegment
FILE_NAME='./sound_of_ball_original.wav'
info = mediainfo(FILE_NAME)
original_sample_rate=int(info['sample_rate'])
sound = AudioSegment.from_file(FILE_NAME)
# sound = AudioSegment.from_file("./output-audio.aac", "aac")
samples = sound.get_array_of_samples()
samples_list=samples.tolist()

abs_samples_list = [abs(ele) for ele in samples_list]
abs_samples_list_pd=pd.Series(abs_samples_list)

#%% slow down to count function
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

#%%
print(sound.frame_count())
print(len(samples_list))
# milliseconds in the sound track
part_file=samples[1200000: 1300000]
short_sound = AudioSegment(part_file.tobytes(), frame_rate=sound.frame_rate,sample_width=sound.sample_width,channels=1)
slow_sound = speed_change(short_sound, 0.1)
#%%
part_samples=slow_sound.get_array_of_samples().tolist()
plt.plot(part_samples)
plt.show()
#%% counting the hits
abs_part_samples=[abs(x) for x in part_samples]
for i in range(90,100):
    percentile=np.percentile(abs_part_samples,i)
    above_list=[1 if x > percentile else 0 for x in abs_part_samples]
    print(percentile, sum(above_list))


# %% where is my cutoff to identify the signal
# 80 th percentile good

window_size=1000
req_percentile=np.percentile(abs_samples_list,80)
max_val=max(abs_samples_list)
# [f(x) if condition else g(x) for x in sequence]
signal_on=[max_val if x >req_percentile else 0 for x in abs_samples_list]
block_signal_on=signal_on.copy()
for i in range(window_size, len(samples_list)-window_size):
    if max_val in signal_on[i-window_size:i+window_size]:
        block_signal_on[i]=max_val
plt.figure(figsize=(10,6),dpi=600)
plt.plot(samples_list)
plt.plot(block_signal_on,'g')
plt.show()
#%% count the bounces how big are windows
count=0
runner=1
block_sizes=[]
block_size=0
block_start=[]
aStart=False
while runner<len(block_signal_on):
    if block_signal_on[runner]!=0 and block_signal_on[runner-1]==0:
        count+=1
        block_sizes.append(block_size)
        block_size=0
        block_start.append(runner)
        aStart=True
    elif block_signal_on[runner]!=0 and block_signal_on[runner-1]!=0:
        block_size+=1
    elif aStart and block_signal_on[runner]==0 and block_signal_on[runner-1]==0:
        aStart=False
        block_start[-1]=(block_start[-1], runner)
        print(runner)
    runner+=1
# print(count, block_sizes)
# print(block_start)
# %%
# 24 bings for first one
s,e = block_start[1]
bingsPerFrame=24/(e-s)
bingsCounts=[]
for i in range(len(block_start)):
    s,e=block_start[i]
    bingsCounts.append((e-s)*bingsPerFrame)
print(bingsCounts)


# %%
