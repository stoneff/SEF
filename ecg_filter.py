import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from Filter import FIR_filter, Matched_filter 

#sdfgsdfgsdfgsdfgsdfgsfdg

# This is for the ECG file recorded before exercise

# read the ECG file with its relevant information
data = np.loadtxt('ecg1.dat')
time_tmp = data[:,0]
time = np.max(time_tmp)/1000
'''keep the frequency as one integer for better analysis'''
fs = int((len(data)-1)/time)

'''
print(data)
print(fs)
print(time)
print(time_tmp)
'''
'''
    for i in range(0,4):
    plot data, compare with the amplitude
    we choose the second channel with highest amplitude
    and so:
'''
data = data[:,2]
''' 
Normalise the data into volt
'''
data = (data-(2**23))*(2.65/(2**24))*(1/500)*1000

# Original ECG: time -- signal

def time_domain_of_ECG(data, time):
    plt.figure(1)
    plt.plot(time, data)
    plt.xlabel("Time, ms")
    plt.ylabel("Voltage, mv") 
    plt.title('Time domain of ECG')
    plt.savefig('figure1.svg')
    plt.show()



# Original ECG: frequency --signal

def frequency_domain_of_ECG(data, fs):
    frequency_domain = np.linspace(0, fs, len(data))
    Fdata = np.fft.fft(data)
    plt.figure(2)
    plt.plot(frequency_domain, abs(Fdata))
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Amplitude")
    plt.title('Frequency_domain_of_ECG')
    plt.savefig('figure2.svg')
    plt.show()


#specify a impulse response remove DC and 50 Hz
''' 
    Note:
    as moving DC when know that frequency resolution = fs/ntaps ,
    because fundamental frequency is 1Hz for our signal,
    and we don't want to remove it. so we specify taps as 2000
    and the resolution is 0.5 HZ. 
    So we won't remove the fudamental frequency
'''

def impulse_response(fs, window_function=None):
    f1 = 0.5
    f2 = 45
    f3 = 55
    ntaps = 2000
    H = np.ones(ntaps)
    index1 = int((f1/fs)*ntaps)
    index2 = int((f2/fs)*ntaps)
    index3 = int((f3/fs)*ntaps)
    H[0: index1] = 0
    H[index2: index3] = 0
    H[ntaps-index1: ntaps] = 0
    H[ntaps-index3: ntaps-index2] = 0
    
    frequency_domain = np.linspace(0, fs, ntaps)
    plt.figure(3)
    plt.plot(frequency_domain,H)
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude = 1')
    plt.title('Ideal frequency response')
    plt.savefig('figure3.svg')    
    
    h = np.fft.ifft(H)
    h = np.real(h)
    
    coeff_h = np.zeros(ntaps)
    coeff_h[0: int(ntaps/2)] = h[int(ntaps/2): ntaps]
    coeff_h[int(ntaps/2): ntaps] = h[0: int(ntaps/2)]
    plt.figure(4)
    plt.plot(coeff_h)
    plt.xlabel('h(n)--n--index number')
    plt.ylabel('Value of coefficients')
    plt.title('Ideal impulse response')
    plt.savefig('figure4.svg')
    plt.show()
    
    if window_function is None:
        return coeff_h
    elif window_function is 'hamming':
        coeff_h= coeff_h*np.hamming(ntaps)
        return coeff_h
    elif window_function is 'hanning':
        coeff_h= coeff_h*np.hanning(ntaps)
        return coeff_h
    elif window_function is 'blackman':
        coeff_h= coeff_h*np.blackman(ntaps)
        return coeff_h
    elif window_function is 'bartlett':
        coeff_h= coeff_h*np.bartlett(ntaps)
        return coeff_h



if __name__ == '__main__':
    
    time_domain_of_ECG(data, time_tmp)
    
    frequency_domain_of_ECG(data, fs)


    # Get the coefficients

    coefficients = impulse_response(fs)

    '''coefficients = impulse_response(fs,'hamming')
       coefficients = impulse_response(fs,'hanning')
       coefficients = impulse_response(fs,'blackman')
       coefficients = impulse_response(fs,'bartlett')
    after trying different window functions, we found it is better not to use it.
    '''

    # FIR_filtered output ECG --time domain
    myfilter = FIR_filter(coefficients)
    output = np.zeros(len(data))
    for i in range(0, len(data)):	
        output[i] = myfilter.dofilter(data[i])
    plt.figure(5)
    plt.plot(time_tmp, output)
    plt.xlabel('Time, ms')
    plt.ylabel('Voltage, mv')
    plt.title('FIR filtered ECG with time domain')
    plt.savefig('figure5.svg')


    # FIR_filtered output ECG --frequency domain
    Foutput = np.fft.fft(output)
    frequency_domain = np.linspace(0,fs,len(data))
    plt.figure(6)
    plt.plot(frequency_domain, abs(Foutput))
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude')
    plt.title('FIR filtered ECG with frequency domain')
    plt.savefig('figure6.svg')
    plt.show()
