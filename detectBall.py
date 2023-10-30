#%%
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statistics
import collections
import pandas as pd
#convert audio to datasegment
sound = AudioSegment.from_file("./output-audio.aac", "aac")
samples = sound.get_array_of_samples()
samples_list=samples.tolist()
starti=21000
endi=22000
abs_samples_list = [abs(ele) for ele in samples_list]
abs_samples_list_pd=pd.Series(abs_samples_list)

#%% slow down to count
#play(sound)  #play sound
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

# play(slow_sound)
#%%
print(sound.frame_count())
print(len(samples_list))
# milliseconds in the sound track
part_file=samples[400000: 500000]
short_sound = AudioSegment(part_file.tobytes(), frame_rate=sound.frame_rate,sample_width=sound.sample_width,channels=1)
slow_sound = speed_change(short_sound, 0.1)
#%%
play(slow_sound)
#%%

300000: 400000
1111111111111111111111111111
#200000: 300000
1111111111111111111111111
# 100000:200000
11111111111111111111111111
111111111111111111111111
111111111111111111111111
part_samples=slow_sound.get_array_of_samples().tolist()
plt.plot(part_samples)
plt.show()
plt.plot(part_file)
plt.show()
#%% counting the hits
abs_part_samples=[abs(x) for x in part_samples]
for i in range(90,100):
    percentile=np.percentile(abs_part_samples,i)
    above_list=[1 if x > percentile else 0 for x in abs_part_samples]
    print(percentile, sum(above_list))


# %%
# 80 th percentile good
req_percentile=np.percentile(abs_samples_list,80)
max_val=max(abs_samples_list)
# [f(x) if condition else g(x) for x in sequence]
signal_on=[max_val if x >req_percentile else 0 for x in abs_samples_list]
block_signal_on=signal_on.copy()
window_size=10
for i in range(window_size, len(samples_list)-window_size):
    if max_val in signal_on[i-window_size:i+window_size]:
        block_signal_on[i]=max_val
plt.plot(samples_list[starti:endi])
plt.plot(signal_on[starti:endi], '--r')
plt.plot(block_signal_on[starti:endi],'g')
plt.show()
#%%
