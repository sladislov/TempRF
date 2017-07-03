
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import math

output_file = open("memes","w")

#enumerate devices
results = SoapySDR.Device.enumerate()
for result in results: print(result)



filterfreq = 160000
rate       = 1e6
frequency  = 868e6+0
length     = 0e0
amplif     = 38
bufflen    = 131072
bandwidth  = 1e6

ifftshit   = 1

#Create HackRF instance
args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)

#Apply settings

sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
sdr.setGain(SOAPY_SDR_RX, 0, amplif)
sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)


#Load sent samples



sent_signal = np.load("OFDM.npy")







#setup a stream (complex integer)
rxStream = sdr.setupStream(SOAPY_SDR_RX, "CS8")
sdr.activateStream(rxStream) #start streaming

dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK

#create a re-usable buffer for rx

buff = np.array([0]*bufflen, np.complex64)
testbuff = np.zeros(bufflen, dtype)  #FUCKING MAGICAL 

bigbuff = np.array([0]*0, dtype)


#receive some samples
for i in range(5):
    sr = sdr.readStream(rxStream, [testbuff], len(testbuff), timeoutUs=1000000)
    print(sr.ret) #num samples or error code
    #print(testbuff)
    print "recieving"
    if (i > 1):
        bigbuff = np.append(bigbuff, testbuff)
print bigbuff.size



#test = signal.fftconvolve(bigbuff,convsig_test[::-1])
testf = np.array([0]*(bigbuff.size/3), np.complex64) #Converts CS8 to CF
testf.real = bigbuff[0:bigbuff.size/3]['re']
testf.imag = bigbuff[0:bigbuff.size/3]['im']

testf2 = np.array([0]*(bigbuff.size/3), np.complex64) #Converts CS8 to CF
testf2.real = bigbuff[bigbuff.size/3:(bigbuff.size/3)*2]['re']
testf2.imag = bigbuff[bigbuff.size/3:(bigbuff.size/3)*2]['im']

testf3 = np.array([0]*(bigbuff.size/3), np.complex64) #Converts CS8 to CF
testf3.real = bigbuff[(bigbuff.size/3)*2:bigbuff.size]['re']
testf3.imag = bigbuff[(bigbuff.size/3)*2:bigbuff.size]['im']


print 'ffting'
##testfft = np.fft.fft((bigbuff['re']), bigbuff.size)
##testfft_scale = np.fft.fftfreq(testfft.size, (1.0/float(rate)))

testfft = np.fft.fft((testf), (testf.size*ifftshit))
testfft_scale = np.fft.fftfreq((testf.size*ifftshit), (1.0/float(rate)))

testfft2 = np.fft.fft((testf2), (testf2.size*ifftshit))
testfft_scale2 = np.fft.fftfreq((testf2.size*ifftshit), (1.0/float(rate)))

testfft3 = np.fft.fft((testf3), (testf3.size*ifftshit))
testfft_scale3 = np.fft.fftfreq((testf3.size*ifftshit), (1.0/float(rate)))

print sent_signal.size
# f, axarr = plt.subplots(4, sharex=False)
# axarr[0].plot(range(bigbuff.size), bigbuff['re'])
# axarr[0].set_title('Recieved Samples')
# axarr[1].plot(range(bigbuff.size), bigbuff['im'])
# axarr[2].plot(testfft_scale, (testfft.real))
# axarr[2].set_title('FFT Recieved R')
# axarr[3].set_title('Frequency Response')
# axarr[3].plot(testfft_scale, testfft.imag)

#plt.show()

#calculate phase and amplitude dude

Amplitude = ((testfft.real*testfft.real)+(testfft.imag*testfft.imag))
Amplitude = np.sqrt(Amplitude)

Amplitude2 = ((testfft2.real*testfft2.real)+(testfft2.imag*testfft2.imag))
Amplitude2 = np.sqrt(Amplitude2)

Amplitude3 = ((testfft3.real*testfft3.real)+(testfft3.imag*testfft3.imag))
Amplitude3 = np.sqrt(Amplitude3)

#Phase = np.arctan((testfft.imag/testfft.real))
Amplitude2[0] = 0

Phase = Amplitude * np.arctan2(testfft.imag, testfft.real)
Phase = (Phase / Phase.max())*np.pi

Phase2 = np.arctan2(testfft2.imag, testfft2.real)

if Phase2.max()>abs(Phase2.min()):
	Phase2 = (Phase2 / Phase2.max())
else:
	Phase2 = (Phase2 / abs(Phase2.min()))


Phase2 = Phase2 * (Amplitude2/Amplitude2.max()) 


Phase3 = Amplitude3 * np.arctan2(testfft3.imag, testfft3.real)
Phase3 = (Phase3 / Phase3.max())*np.pi


# detect carriers


Ampdetect = Amplitude2.max()/2
CarrierPOS = np.array([0]*0, int)




#Fill an array with location

for i in range (Amplitude2.size):
	if (Amplitude2[i] > (Ampdetect/2)):
		CarrierPOS = np.append(CarrierPOS,i)


phasefault = Phase2[CarrierPOS]      #Correct phase?
# for i in range (phasefault.size):
# 	if phasefault[i] < 0:
# 		 phasefault[i] = 1 - phasefault[i] #????

deltaT = (phasefault * (0.5/(CarrierPOS*(rate/bufflen)))) #Calc time error


deltaS = deltaT/(1/rate)



if (deltaT.max() > abs(deltaT.min())):
	deltaTmax = deltaT.max()
else:
	deltaTmax = deltaT.min()

deltaTmax = deltaT[0]



samplecorr = deltaTmax/(1/rate)

samplecorr2 = -int(samplecorr)

# samplecorr = 



print (CarrierPOS*(rate/bufflen))
print phasefault
print deltaT
print deltaS
print samplecorr2

#samplecorr = 227
#samplecorr = 0

# try corrected samples
# samplecorr = 113

testf4 = np.array([0]*(bigbuff.size/3), np.complex64) #Converts CS8 to CF
testf4.real = bigbuff[(samplecorr2+(bigbuff.size/3)):(samplecorr2+((bigbuff.size/3)*2))]['re']
testf4.imag = bigbuff[(samplecorr2+(bigbuff.size/3)):(samplecorr2+((bigbuff.size/3)*2))]['im']





testfft4 = np.fft.fft((testf4), (testf4.size*ifftshit))
testfft_scale4 = np.fft.fftfreq((testf4.size*ifftshit), (1.0/float(rate)))


# for i in range(testfft4.size):
# 	testfft4 = testfft4*(np.cos(2*np.pi*testfft_scale4[i]+((samplecorr-samplecorr2)*(1/rate)))+(1j*np.sin(2*np.pi*testfft_scale4[i]+((samplecorr-samplecorr2)*(1/rate)))))
correctionfactor = samplecorr+samplecorr2

correctionfactor = -correctionfactor 

print correctionfactor

testfft4[CarrierPOS] = testfft4[CarrierPOS] * (np.cos(2*np.pi*testfft_scale4[CarrierPOS]*((correctionfactor)*(1/rate)))+(1j*np.sin(2*np.pi*testfft_scale4[CarrierPOS]*((correctionfactor)*(1/rate)))))



print np.cos(2*np.pi*testfft_scale4[CarrierPOS]*((correctionfactor)*(1/rate)))

print 1j*np.sin(2*np.pi*testfft_scale4[CarrierPOS]*((correctionfactor)*(1/rate)))




Amplitude4 = ((testfft4.real*testfft4.real)+(testfft4.imag*testfft4.imag))
Amplitude4 = np.sqrt(Amplitude4)

Amplitude4[0] = 0

Phase4 = np.arctan2(testfft4.imag, testfft4.real)

if Phase4.max()>abs(Phase4.min()):
	Phase4 = (Phase4 / Phase4.max())
else:
	Phase4 = (Phase4 / abs(Phase4.min()))


Phase4 = Phase4 * (Amplitude4/Amplitude4.max()) 

Phase4[0] = 0


# f, axarr = plt.subplots(8, sharex=True)
# axarr[0].plot(testfft_scale, Amplitude)
# axarr[0].set_title('Amplitude Symbol 1')
# axarr[1].plot(testfft_scale, Phase)
# axarr[1].set_title('Phase Symbol 1')
# axarr[2].plot(testfft_scale2, Amplitude2)
# axarr[2].set_title('Amplitude Symbol 2')
# axarr[3].plot(testfft_scale2, Phase2)
# axarr[3].set_title('Phase Symbol 2')
# axarr[4].plot(testfft_scale3, Amplitude3)
# axarr[4].set_title('Amplitude Symbol 3')
# axarr[5].plot(testfft_scale3, Phase3)
# axarr[5].set_title('Phase Symbol 3')
# axarr[6].plot(testfft_scale4, Amplitude4)
# axarr[6].set_title('99 Problems')
# axarr[7].plot(testfft_scale4, Phase4)
# axarr[7].set_title('But OFDM aint one')

f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(testfft_scale, Amplitude2)
# axarr[0].plot(testfft_scale, testfft4.real)
axarr[0].set_title('Amplitude Symbol 2')
axarr[1].plot(testfft_scale, Phase2)
# axarr[1].plot(testfft_scale, testfft4.imag)
axarr[1].set_title('Phase Symbol 2')
axarr[2].plot(testfft_scale4, Amplitude4)
axarr[2].set_title('99 Problems')
axarr[3].plot(testfft_scale4, Phase4)
axarr[3].set_title('But OFDM aint one')


plt.show()

#shutdown the stream
sdr.deactivateStream(rxStream) #stop streaming
sdr.closeStream(rxStream)