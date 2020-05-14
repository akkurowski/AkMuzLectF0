import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os 
from scipy.signal import butter, lfilter
from collections import Counter

def butter_lowpass_filter(data, normalCutoff, order=4):
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    y = lfilter(b, a, data)
    return y

def nice_plot_timedomain(vector_timedomain, fs):
    time_vector = np.linspace(0,1,len(vector_timedomain))*len(vector_timedomain)/fs
    plt.plot(time_vector,vector_timedomain)
    plt.xlabel('time [s]')
    plt.ylabel('instantaneous value [-]')

def nice_plot_specdomain(filtered_spectrum, fs, mode='log'):
    frequency_vector = np.linspace(0,1,len(filtered_spectrum))*fs*0.5
    if mode == 'log':
        plt.semilogy(frequency_vector,filtered_spectrum)
    elif mode == 'lin':
        plt.plot(frequency_vector,filtered_spectrum)
    else:
        raise RuntimeError('Choose correct plot type')
        
    plt.grid()
    plt.xlabel('frequency [Hz]')
    plt.ylabel('amplitude spectrum [-]')

def allwithall_difference(frequencies):
    differences = []
    for i in range(0,len(frequencies)):
        for j in range(0,len(frequencies)):
            if i==j:
                continue
            if i>j:
                continue
            differences.append(np.abs(frequencies[i] - frequencies[j]))
    return sorted(differences)

os.system('cls')

fs, samples             = wavfile.read('fragment.wav')
samples                 = samples/np.max(np.abs(samples))

frame_fft               = np.fft.fft(samples)
amplitude_spectrum      = np.abs(frame_fft)

desired_spectrum_length = len(amplitude_spectrum)//2
amplitude_spectrum      = amplitude_spectrum[0:desired_spectrum_length]
frequencies             = np.linspace(0,fs/2,desired_spectrum_length)

filtered_spectrum       = butter_lowpass_filter(amplitude_spectrum, 0.25, order=1)
trend_spectrum          = butter_lowpass_filter(amplitude_spectrum, 0.002, order=1)

detrended_spectrum      = filtered_spectrum - trend_spectrum

binary_spectrum         = np.zeros_like(detrended_spectrum)
binary_spectrum[detrended_spectrum>17] = 1

differentiated_spectrum = np.diff(binary_spectrum)

frequencies         = frequencies[1:]

left_frequencies    = frequencies[differentiated_spectrum==1]
right_frequencies   = frequencies[differentiated_spectrum==-1]

frequencies_mtx = np.matrix([left_frequencies,right_frequencies])
central_frequencies = np.mean(frequencies_mtx,axis=0)

diff_harms = allwithall_difference(central_frequencies.tolist()[0])

print(diff_harms)

# plt.figure()
# nice_plot_specdomain(amplitude_spectrum, fs, mode='log')

# plt.figure()
# nice_plot_specdomain(filtered_spectrum, fs, mode='log')

# plt.figure()
# nice_plot_specdomain(trend_spectrum, fs, mode='log')

# plt.figure()
# nice_plot_specdomain(differentiated_spectrum, fs, mode='lin')

# plt.figure()
# plt.plot(frequencies,amplitude_spectrum)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('amplitude [-]')

plt.show()