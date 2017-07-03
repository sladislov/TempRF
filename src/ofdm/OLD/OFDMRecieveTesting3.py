import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import math

output_file = open("memes","w")

dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK

filterfreq = 160000
rate       = 1e6
frequency  = 868e6+0
length     = 0e0
amplif     = 30
bufflen    = 131072
bandwidth  = 20e6

ifftshit   = 1

def init_sdr ():

	#enumerate devices
	results = SoapySDR.Device.enumerate()
	for result in results: print(result)

	#Create HackRF instance
	args = dict(driver="hackrf")
	sdr = SoapySDR.Device(args)

	#Apply settings

	sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
	sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
	sdr.setGain(SOAPY_SDR_RX, 0, amplif)
	sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)


	

	return sdr

def recieve_samples(cycle, sdr):
	#create a re-usable buffer for rx
	buff = np.array([0]*bufflen, np.complex64)
	testbuff = np.zeros(bufflen, dtype)  #FUCKING MAGICAL 
	bigbuff = np.array([0]*0, dtype)

	#setup a stream (complex integer)
	rxStream = sdr.setupStream(SOAPY_SDR_RX, "CS8")
	sdr.activateStream(rxStream) #start streaming

	#receive some samples
	for i in range(cycle):
	    sr = sdr.readStream(rxStream, [testbuff], len(testbuff), timeoutUs=1000000)
	    print(sr.ret) #num samples or error code
	    #print(testbuff)
	    print "recieving"
	    if (i > 1):
	        bigbuff = np.append(bigbuff, testbuff)
	return bigbuff

def detect_carrier(Amplitude):
	Amplitude[0] = 0
	Ampdetect = Amplitude.max()/2
	CarrierPOS = np.array([0]*0, int)

	for i in range (Amplitude.size):
		if (Amplitude[i] > (Ampdetect/2)):
			CarrierPOS = np.append(CarrierPOS,i)
	return CarrierPOS

def return_amp_phase(cffft):
	Amplitude = ((cffft.real*cffft.real)+(cffft.imag*cffft.imag))
	Amplitude = np.sqrt(Amplitude)

	#Amplitude[0] = 0

	Phase = np.arctan2(cffft.imag, cffft.real)

	# if Phase.max()>abs(Phase.min()):
	# 	Phase = (Phase / Phase.max())
	# else:
	# 	Phase = (Phase / abs(Phase.min()))
	Phase = np.angle(cffft)/(np.pi)


	if Amplitude.max()>abs(Amplitude.min()):
		Phase = Phase * (Amplitude/Amplitude.max()) 
	else:
		Phase = Phase * (Amplitude/abs(Amplitude.min())) 



	
	#Phase[0] = 0

	return {'AMP':Amplitude, 'PHASE':Phase}

def calc_IQFFT(fbuff): ##Give this CF thanks
	fbuff = cs_to_cf(fbuff)
	cffft = np.fft.fft((fbuff), (fbuff.size*ifftshit))
	cffft_scale = np.fft.fftfreq((fbuff.size*ifftshit), (1.0/float(rate)))

	return {'IQFFT':cffft, 'IQFFT_SCALE':cffft_scale}

def cs_to_cf(cs): #????

	CF = np.array([0]*(cs.size), np.complex64) #Converts CS8 to CF
	CF.real = bigbuff[0:cs.size]['re']
	CF.imag = bigbuff[0:cs.size]['im']

	return CF



sdr = init_sdr()

bigbuff = recieve_samples(3, sdr)

Init_FFT = calc_IQFFT(bigbuff)

Init_PhaseAmp = return_amp_phase(Init_FFT['IQFFT'])



Init_FFT['IQFFT'] *= np.exp(-1j*2*np.pi*(Init_PhaseAmp['PHASE'][8192]/(rate*bufflen)*Init_FFT['IQFFT_SCALE']))


Init_PhaseAmp2 = return_amp_phase(Init_FFT['IQFFT'])

f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(Init_FFT['IQFFT_SCALE'], Init_PhaseAmp['AMP'])
axarr[1].plot(Init_FFT['IQFFT_SCALE'], Init_PhaseAmp['PHASE'])
axarr[2].plot(Init_FFT['IQFFT_SCALE'], Init_PhaseAmp2['AMP'])
axarr[3].plot(Init_FFT['IQFFT_SCALE'], Init_PhaseAmp2['PHASE'])
plt.show()

print bigbuff