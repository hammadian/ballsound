#%%
from pydub import AudioSegment
import numpy as np
import os
import librosa
from scipy.signal import argrelmax
import pandas as pd
from pydub import  silence

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

#%%
print(np.median(samples_list))
print(np.mean(samples_list))
print(np.percentile(samples_list,[70,80,90]))
#plt.plot(samples_list)
#%%
one_sample=samples[100000:180000]
short_sound = AudioSegment(one_sample.tobytes(), frame_rate=sound.frame_rate,sample_width=sound.sample_width,channels=1)
play(short_sound)
file_handle = short_sound.export("short_sound_output.wav", format="wav")

# %%
rosa_sound, sr=librosa.load(FILE_NAME, sr=int(original_sample_rate))
ft=np.fft.fft(rosa_sound)
ft_mag=np.abs(ft)
plt.plot(ft)
plt.show()
plt.plot(ft_mag)
# %%
f_ratio=0.2

freq_x=np.linspace(0, original_sample_rate, len(ft_mag))
num_freq_bins=int(len(freq_x)* f_ratio)
# plt.plot(freq_x[:num_freq_bins], ft_mag[:num_freq_bins])
# plt.show()
plt.figure
plt.plot([x if x > 3000 else 0 for x in ft_mag[:2000]])
ft_mag_list=ft_mag.tolist()
np.argmax(ft_mag_list)
# %%
frequencey_with_max_magnitude=freq_x[ft_mag_list.index(max(ft_mag_list))]
new = sound.low_pass_filter(int(frequencey_with_max_magnitude))
new_short_lpf=short_sound.low_pass_filter(int(frequencey_with_max_magnitude))
plt.figure(figsize=[200,20])
short_sound_list=short_sound.get_array_of_samples()
new_short_lpf_list=new_short_lpf.get_array_of_samples()
plt.plot(short_sound_list,'-')
plt.plot(new_short_lpf_list,'o',color='r')
powz=3
short_sound_list_powered=list(map(lambda x:pow(x,powz),short_sound_list))
short_sound_df=pd.Series(short_sound_list_powered)
short_sound_list_powered_smooth1=short_sound_df.rolling(500).mean()
plt.plot(short_sound_list_powered_smooth1,'+',color='g')
short_sound_list_powered_smooth2=short_sound_list_powered_smooth1.rolling(500).mean()
plt.plot(short_sound_list_powered_smooth1,'+',color='black')

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
slow_sound_lpf = speed_change(new_short_lpf, 0.1)
# play(slow_sound+20)
# %%
plt.figure(figsize=(12,7))
slow_sound_lpf_list=list(slow_sound_lpf.get_array_of_samples())
plt.plot(slow_sound)
# %%
file_handle = slow_sound_lpf.export("slow_sound_output.wav", format="wav")
# %%
slow_sound_list_df=pd.Series(slow_sound_lpf_list)
smooth1_slow_sound_list=slow_sound_list_df.rolling(10000).max()
smooth2_slow_sound_list=slow_sound_list_df.rolling(10000).min()
plt.figure(figsize=[15,5])
plt.plot(smooth1_slow_sound_list)
plt.plot(smooth2_slow_sound_list)
# plt.plot(slow_sound_list)
plt.show()

# %%
from scipy.signal import find_peaks
# peaks, _ = find_peaks(slow_sound_lpf_list, height=threshold, distance=min_distance_between_peaks)
maxAmp=max(slow_sound_list_df)
peaks, _ = find_peaks(slow_sound_lpf_list, height=0.5*maxAmp)
# %%
len(peaks)
# %%
# a positive peak is max between two negatives
peakpeak=[0]*len(slow_sound_list_df)
positives=[]
negatives=[]
for i in peaks:
    if slow_sound_list_df[i]<0:
        negatives.append((i, slow_sound_list_df[i]))
        # mark the peak
        if positives:
            aPair=max(positives, key=lambda item: item[1])
            print(aPair)
            peakpeak[aPair[0]]=aPair[1]
        positives=[]
    else:
        positives.append((i, slow_sound_list_df[i]))
        #negs
        if negatives:
            aPair=min(negatives, key=lambda item: item[1])
            peakpeak[aPair[0]]=aPair[1]
        negatives=[]
plt.figure(figsize=(20,7))
plt.plot(slow_sound_lpf_list)
plt.plot(peaks,slow_sound_list_df[peaks],'o', color='g')
plt.plot(peakpeak,'o',color='r')
sum([1 if i>0 else 0 for i in peakpeak])


# %%
peaks
# %% fitting sine wave
import numpy, scipy.optimize

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}
