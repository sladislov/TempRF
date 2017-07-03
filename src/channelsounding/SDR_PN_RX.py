########################################################################
## HackRF Reciever and Saver
########################################################################
## Do not comment about format you will trigger me
########################################################################
##
## SDR-R 
## Recieves HackRF data and saves it to file
## Copyright Laszlo Olk 2018-1
##
##
########################################################################
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
import scipy
from scipy import signal

output_file = open("memes","w")

#enumerate devices
results = SoapySDR.Device.enumerate()
for result in results: print(result)



filterfreq = 160000
rate       = 1e6
frequency  = 868e6
length     = 0e0
amplif     = 48
bufflen    = 131072
bandwidth  = 1e6


#Create HackRF instance
args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)

#Apply settings

sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
sdr.setGain(SOAPY_SDR_RX, 0, amplif)
sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)


#Load sent samples

pn_sample = np.load("FFTPN_filter.npy")

sent_signal = np.load("PN_filter.npy")

convsig_test   = np.load("test.npy")

print("\nFrequency set to:") + str(sdr.getFrequency(SOAPY_SDR_RX, 0))



#setup a stream (complex integer)
rxStream = sdr.setupStream(SOAPY_SDR_RX, "CS8")
sdr.activateStream(rxStream) #start streaming

print "\n"

clksrc = sdr.getClockSource()
clklist = sdr.listClockSources()
snslist = sdr.listSensors()

print clklist
print snslist

sdr.setClockSource("external")

dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK

#create a re-usable buffer for rx

buff = np.array([0]*bufflen, np.complex64)
testbuff = np.zeros(bufflen, dtype)  #FUCKING MAGICAL 

bigbuff = np.array([0]*0, dtype)


#receive some samples
for i in range(3):
    sr = sdr.readStream(rxStream, [testbuff], len(testbuff), timeoutUs=1000000)
    print(sr.ret) #num samples or error code
    #print(testbuff)
    print "recieving"
    #if (i > 1):
    #    bigbuff = np.append(bigbuff, testbuff.view(np.int8).astype(np.float32).view(np.complex64))
    if (i > 1):
        bigbuff = np.append(bigbuff, testbuff)
        #print bigbuff
    #numpy.savetxt('outfile.txt', buff)
print bigbuff.size



#test = signal.fftconvolve(bigbuff,convsig_test[::-1])
testf = np.array([0]*bigbuff.size, np.complex64) #Converts CS8 to CF
testf.real = bigbuff['re']
testf.imag = bigbuff['im']

testf2 = np.array([0]*sent_signal.size, np.complex64)
testf2.real = sent_signal['re']
testf2.imag = sent_signal['im']

test = signal.fftconvolve(testf, testf2[::-1]) #Correlate recieve and sent
test2 = signal.fftconvolve(testf2, testf2[::-1])

print 'ffting'
##testfft = np.fft.fft((bigbuff['re']), bigbuff.size)
##testfft_scale = np.fft.fftfreq(testfft.size, (1.0/float(rate)))

testfft = np.fft.fft((testf), (testf.size*32))
testfft_scale = np.fft.fftfreq((testf.size*32), (1.0/float(rate)))

freqres = np.fft.fft(test, test.size)
freqres_scale = np.fft.fftfreq(test.size, (1.0/float(rate)))

print sent_signal.size
f, axarr = plt.subplots(6, sharex=False)
axarr[0].plot(range(bigbuff.size), bigbuff['re'])
axarr[0].set_title('Recieved Samples')
axarr[1].plot(range(bigbuff.size), bigbuff['im'])
axarr[2].set_title('Correlation')
axarr[2].plot(range(test.size), test.real)
axarr[3].set_title('Autocorrelation Input')
#axarr[3].plot(range(bigbuff.size), (bigbuff['re'] + bigbuff['im']))
axarr[3].plot(range(test2.size), test2.real)
axarr[4].plot(testfft_scale, testfft.real)
axarr[4].set_title('FFT Recieved R')
axarr[5].set_title('Frequency Response')
#axarr[5].plot(freqres_scale, freqres)
axarr[5].plot(testfft_scale, testfft.imag)

plt.show()


    
#shutdown the stream
sdr.deactivateStream(rxStream) #stop streaming
sdr.closeStream(rxStream)
