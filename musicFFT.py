import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile # get the api
import numpy as np
import math

def roundup(x, interval):
    return int(math.ceil(x/interval)) * interval

def stereoToMono(audiodata):
    newaudiodata = audiodata.sum(axis=1) / 2
    return np.array(newaudiodata, dtype='int16')


if __name__ == '__main__':

    #IMPORT .wav file and change from stereo to mono
    wav_sampleRate, wav_data = wavfile.read('disconnect_me.wav') #Get WAV Sample Rate (samples/sec) and Data

    #Make a stereo track mono
    if np.array(wav_data).ndim == 2:
        wav_data = stereoToMono(wav_data) #Stereo to Mono (Combine Left and Right channel of WAV data)

    #Discrete Time Signal
    ampl = np.abs(wav_data) #signal amplitude vector
    fs = wav_sampleRate #signal sample rate
    N = wav_data.size #number of samples in wav_data
    #t = np.arange(N)/fs #time of each sample vector
    t = np.linspace(0, N/fs, N) #time of each sample vector

    #Fast Fourier Transform
    dft = np.fft.rfft(wav_data) #DFT vector (Complex Numbers)
    dft_angle = np.angle(dft)   #DFT Angle (Radians)
    dft_phase = np.unwrap(dft_angle) #DFT Phase (Radians)
    dft_magnitude = np.abs(dft)/N*2 #Scaled DFT Magnitude (Amplitude)
    dt = 1/float(fs) #sample spacing (inverse of sample rate)
    f = np.fft.rfftfreq(N,dt) #DFT sample frequency vector

    ####################################
    ##PLOTTING
    ####################################

    #create 3x1 sub plots
    gs = gridspec.GridSpec(3,1)

    #Amplitude vs Time
    plt.figure()
    sp1 = plt.subplot(gs[0,:]) # row 0, span all columns
    sp1.set_title('Signal')
    sp1.set_ylabel('Amplitude') # x(t)
    sp1.set_xlabel('Time (s)')
    sp1.grid('on')
    plt.plot(t,ampl) #(time, amplitude)

    #Amplitude vs Frequency
    sp2 = plt.subplot(gs[1,:]) # row 1, span all columns
    sp2.set_title('Amplitude Spectrum')
    sp2.set_ylabel('Amplitude') # |X[f]|
    sp2.set_xlabel('Frequency (Hz)')
    sp2.set_xscale('log')
    sp2.grid('on')
    plt.plot(f,dft_magnitude) #(frequency, amplitude)

    #Phase vs Frequency
    sp3 = plt.subplot(gs[2,:]) # row 2, span all columns
    sp3.set_title('Phase Spectrum')
    sp3.set_ylabel('Phase (radians)') # ∠X(k)
    sp3.set_xlabel('Frequency (Hz)')
    sp3.set_xscale('log')
    sp3.grid('on')
    # plt.plot(f, dft_phase); #(frequency, phase)
    plt.phase_spectrum(wav_data, fs)


    plt.tight_layout()
    plt.show()
    #plt.savefig('AudioFFT.png')