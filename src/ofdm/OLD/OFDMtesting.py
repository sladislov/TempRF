import numpy as np
import time
import math
import commpy
import matplotlib.pyplot as plt
import scipy
import os


def OFDM_Encoder(x, df, sr, p):
    i = 0
    j = 0
    interpol = 128
    print "a"
##    Output = np.zeros((sr/df), np.complex64)
##    const = sr/df
##    while (len(x) > i):
##        while (Output.size > j):
##            Output[j] = Output[j] + math.cos(2.0*math.pi*(float(j)/((const/(i+1))))+(x[i]+0.5)*math.pi*(1.0/2.0))
##            #print j
##            j = j + 1
##        j = 0
##        #print (x[i]+0.5)*math.pi*(1.0/2.0)
##        i = i + 1
##    return Output
    z = np.zeros(((len(x)*interpol)+0), np.complex64)
    z = np.fft.fftshift(z)
    while i < len(x)/2:
        if x[i*2] > 0:
            z.real[i*interpol] = 1/math.sqrt(2)
        elif x[i*2] < 0:
            z.real[i*interpol] = -1/math.sqrt(2)

        #print z.real[i]
            
        if x[(i*2)+1] > 0 :
            z.imag[i*interpol] = 0/math.sqrt(2)
        else:
            z.imag[i*interpol] = -0/math.sqrt(2)
        i = i + 1
    z = np.roll(z, 0*interpol)
    print "z"
    return z
    

    

### Dicking around with OFDM Theory
sr = 1.0e6

x = np.array([0]*1024, np.int8)


#x[164:1024:64] = -1
#x[64:1024:128] = 1

x[128:1024:128] = 1
#x[170] = -1
#x[340] = -1
#x[65:513:4] = 1

print len(x)
df = 50.0
p = "1111"

a = OFDM_Encoder(x, df, sr, p)

print a.size
    
z = np.fft.ifft(a, len(a))

print "done"

a_scale = np.fft.fftfreq(a.size, (1.0/float(sr)))


# f, axarr = plt.subplots(5, sharex=False)
# axarr[0].plot(a_scale, a.real)
# axarr[0].set_title('FFT signal REAL')
# axarr[1].plot(a_scale, a.imag)
# axarr[1].set_title('FFT signal IMAG')
# axarr[2].plot(range(z.size), z.real)
# axarr[2].set_title('signal REAL')
# axarr[3].plot(range(z.size), z.imag)
# axarr[3].set_title('signal IMAG')
# axarr[4].plot(range(a.size), np.fft.fftshift(a.real))

i=0
dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK
CS8_Samps = np.zeros(len(z), dtype)


if z.real.max()>abs(z.real.min()):
    z.real *= (126 / z.real.max())
else:
    z.real *= (126 / abs(z.real.min()))

if z.imag.max()>abs(z.imag.min()):
    z.imag *= (126 / z.imag.max())
else:
    z.imag *= (126 / abs(z.imag.min()))

# # YOU ARE RETARDED
# z.real *= 126.0/z.real.max()
# z.real *= -126.0/z.real.min()

# z.imag *= 126.0/z.imag.max()
# z.imag *= -126.0/z.imag.min()
# #FUCKING RETARD



CS8_Samps['re'] = (z.real.astype(np.int8))
CS8_Samps['im'] = (z.imag.astype(np.int8))

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(range(CS8_Samps.size), CS8_Samps['re'])
axarr[0].set_title('FFT signal REAL')
axarr[1].plot(range(z.size), z.real)
axarr[1].set_title('FFT signal IMAG')




x = z
x.real = CS8_Samps['re']
x.imag = CS8_Samps['im']

y = np.fft.fft(x, len(x))
y_scale = np.fft.fftfreq(y.size, (1.0/float(sr)))




Amplitude = ((y.real*y.real)+(y.imag*y.imag))
Amplitude = np.sqrt(Amplitude)

#mplitude[0] = 0

Phase = np.arctan2(y.imag, y.real)

if Phase.max()>abs(Phase.min()):
    Phase = (Phase / Phase.max())
else:
    Phase = (Phase / abs(Phase.min()))

if Amplitude.max()>abs(Amplitude.min()):
    Phase = Phase * (Amplitude/Amplitude.max()) 
else:
    Phase = Phase * (Amplitude/abs(Amplitude.min())) 




f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(y_scale, y.real)
axarr[0].set_title('Reality')
axarr[1].plot(y_scale, y.imag)
axarr[1].set_title('My hopes and dreams')
axarr[2].plot(y_scale, Amplitude)
axarr[3].plot(y_scale, Phase)

plt.show()
np.save('OFDM', CS8_Samps)

print CS8_Samps.size

import OFDMSendTesting.py




