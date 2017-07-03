




########################################################################
## PN generator and filterer 
########################################################################
## Do not comment about format you will trigger me
########################################################################
##
## SDR_PN_SEQ
## PN sequence creator and filterer
## Copyright Laszlo Olk 2018-1
##
##
########################################################################

import numpy as np
import time
import math
import commpy
import matplotlib.pyplot as plt
from scipy import signal
import scipy

###shhhhhh
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""

    return int(np.ceil(np.log2(np.abs(x))))

def freq_shift(x, f_shift, dt):
    """
    Shift the specified signal by the specified frequency.
    """

    # Pad the signal with zeros to prevent the FFT invoked by the transform from
    # slowing down the computation:
    N_orig = len(x)
    N_padded = 2**nextpow2(N_orig)
    t = np.arange(0, N_padded)
    return (scipy.signal.hilbert(np.hstack((x, np.zeros(N_padded-N_orig, x.dtype))))*np.exp(2j*np.pi*f_shift*dt*t))[:N_orig].real



filterfreq    =     6.30e4                ##Filtering frequency, filters to the left and right side of N
#filterfreq    = 10e6
rate          =     1e6                ##Sample rate
dt            =   1/rate
interpolation =    3                ##Interpolation required? 1 = no interpolation, 2 = one zero, 3 = two zeroes, etc
pn_order      =    18                  ##Pn order, 2^n
pn_seed       = "100010101001101111"     ##Pn seed
pn_mask       = "000000000010000001"     ##Pn mask # "100001000100010001" 3 peaks #
pn_length     = int((math.pow(2, pn_order)-1)*interpolation)        ##Pn length, Should be ((2^pn_order)-1)*interpolation. MAKE SURE THIS IS NOT A PRIME, ITS A CRIME 
sample_rate   =     1e6                ##Unused
start_time    = time.time()
print start_time
print pn_length
#sample arrays

dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC

CS8_Samps = np.zeros(pn_length, dtype)

print CS8_Samps

samples       = np.array([0]*(pn_length), np.complex64)
samplesFFT    = np.array([0]*(pn_length), np.complex64)
samplesFFTfilt= np.array([0]*(pn_length), np.complex64)
filt_samples  = np.array([0]*(pn_length), np.complex64)


#open file
output  = open("output.meme", "w")

#generate pn sequence

print "generating pn sequence" + " elapsed time:" + str((time.time() - start_time))

pn_sequence   = commpy.pnsequence(pn_order, pn_seed, pn_mask, (pn_length/interpolation))

#put pn sequence in sample array
i = 0

while ( i < (pn_length)):
    if (i % interpolation == 0):
        if (pn_sequence[(i/interpolation)-1] == 1):
        #if ((i/interpolation)%2 == 0): #enable this for square wave
            samples[i] = 1
        else:
            samples[i] = -1
    i = i + 1

##while ( i < pn_length):
##    samples[i] = math.sin((float(2)*math.pi*(float(i)/float(2048))))
##    i = i + 1



#FFT the samples
print "performing first fft" + "  time:" + str((time.time() - start_time))
samplesFFT = np.fft.fft(samples, pn_length)
samplesFFT_scale = np.fft.fftfreq(samplesFFT.size, (1.0/float(sample_rate)))
samplesFFTfilt = samplesFFT.copy()
#Filtering FFT'd samples
print "filtering" + " elapsed time:" + str((time.time() - start_time))
samplesFFTfilt.real[samplesFFT_scale<-filterfreq] = 0
samplesFFTfilt.real[samplesFFT_scale>filterfreq]  = 0
samplesFFTfilt.imag[samplesFFT_scale<-filterfreq] = 0
samplesFFTfilt.imag[samplesFFT_scale>filterfreq]  = 0
##
##samplesFFTfilt = np.fft.fftshift(samplesFFTfilt)
##samplesFFTfilt = np.roll(samplesFFTfilt, int(0))
##samplesFFTfilt = np.fft.ifftshift(samplesFFTfilt)
##



#ifft the Filtered FFT'd sample
print "IFFT'ing the filtered samples" + " time:" + str((time.time() - start_time))
filt_samples = np.fft.ifft(samplesFFTfilt)

filt_samples = freq_shift(filt_samples.real, 2000, dt)

print "Plotting your demise (and the signals)" + " time:" + str((time.time() - start_time))
f, axarr = plt.subplots(9, sharex=False)
axarr[0].scatter(samplesFFT_scale, samplesFFT.real)
axarr[0].set_title('Original Real')
##axarr[0].scatter(range(a.size), a.real)
axarr[1].scatter(samplesFFT_scale, samplesFFTfilt.real)
axarr[1].set_title('Copy Real')
axarr[2].scatter(samplesFFT_scale, samplesFFT.imag)
axarr[2].set_title('Original Imag')
axarr[3].scatter(samplesFFT_scale, samplesFFTfilt.imag)
axarr[3].set_title('Copy Imag')
axarr[4].plot(range(samples.size), samples.real)
axarr[4].set_title('Original Signal Real')
axarr[5].plot(range(samples.size), filt_samples.real)
axarr[5].set_title('IFFT Signal Real')
axarr[6].plot(range(samples.size), samples.imag)
axarr[6].set_title('Original Signal Imag')
axarr[7].plot(range(samples.size), filt_samples.imag)
axarr[7].set_title('IFFT Signal Imag')
print "done plotting" + "  time:" + str((time.time() - start_time))


plt.show()




output.write("Interpolation = " + str(interpolation) + '\n')
output.write("Pn_order      = " + str(pn_order) + '\n')
output.write("Pn_seed       = " + str(pn_seed) + '\n')
output.write("Pn_mask       = " + str(pn_mask) + '\n')
output.write("Pn_length     = " + str(pn_length) + '\n')

#output.write(filt_samples)

#scaling signal
print "scaling" + " elapsed time:" + str((time.time() - start_time))
i = 0


max_scale = 127.0/(np.amax(filt_samples))
while (i <(pn_length)):                         ## PRE-SCALE the floats for conversion to CS8
    filt_samples[i] = filt_samples[i] * max_scale
    i = i + 1

    


CS8_Samps['re'] = (filt_samples.real.astype(np.int8))
CS8_Samps['im'] = (filt_samples.imag.astype(np.int8))

memes = signal.fftconvolve(filt_samples, filt_samples[::-1])

a = np.fft.fft(filt_samples, filt_samples.size)


b = a*a[::-1]

b = np.fft.ifft (b, b.size)



f, axarr = plt.subplots(4, sharex=True)
##axarr[0].plot(range(CS8_Samps['re'].size), CS8_Samps['re'])
##axarr[0].set_title('CS8.real')
##axarr[1].plot(range(CS8_Samps['re'].size), filt_samples.real)
##axarr[1].set_title('CF64.real')
##axarr[2].plot(range(CS8_Samps['re'].size), CS8_Samps['im'])
##axarr[2].set_title('CS8.imag')
##axarr[3].plot(range(CS8_Samps['re'].size), filt_samples.imag)
##axarr[3].set_title('CF64.imag')
axarr[1].plot(range(b.size), b)
axarr[2].plot(range(a.size), a)
axarr[3].plot(range(memes.size), memes)
axarr[3].set_title('AutoCorr')

plt.show()


np.save("test", filt_samples)
np.save("PN_filter", CS8_Samps)
np.save("FFT" + "PN_filter", samplesFFTfilt)

output.close()




