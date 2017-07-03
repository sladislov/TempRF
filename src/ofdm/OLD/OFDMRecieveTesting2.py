# OFDMRecieveTesting2
# Minder gebeun dan 1
# Even kloten
# 
# Laszlo Olk 2017

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

def cf_to_cs(cf):  #DO NOT USE THIS FOR AMPLITUDE SIGNALS i think it messes up NEVERMIND i FIXED IT 

	CS = np.zeros(len(cf), dtype)

	if cf.real.max()>abs(cf.real.min()):
	    cf.real *= (126 / cf.real.max())
	else:
	    cf.real *= (126 / abs(cf.real.min()))

	if cf.imag.max()>abs(cf.imag.min()):
	    cf.imag *= (126 / cf.imag.max())
	else:
	    cf.imag *= (126 / abs(cf.imag.min()))

	CS['re'] = (cf.real.astype(np.int8))
	CS['im'] = (cf.imag.astype(np.int8))

	return CS





def cs_to_cf(cs): #????

	CF = np.array([0]*(cs.size), np.complex64) #Converts CS8 to CF
	CF.real = bigbuff[0:cs.size]['re']
	CF.imag = bigbuff[0:cs.size]['im']

	return CF



def fix_symbol(buff):                                                                 #IN COMPLEX FLOATS YOU TRIPLE 

	Acquibuff = buff[buff.size/3:((buff.size/3)*2)]
	FFT_Acquibuff = calc_IQFFT(Acquibuff)                                              #get fft
	AP_Acquibuff = return_amp_phase(FFT_Acquibuff['IQFFT'])							  #get amp n phase
	CarrierPOS = detect_carrier(AP_Acquibuff['AMP'])	

	#print CarrierPOS

	PilotPOS = np.array([8192,24576,40960,57344])

	print CarrierPOS
	print PilotPOS

	print (rate/Acquibuff.size)
													  #detect carriers

	deltaT = (AP_Acquibuff['PHASE'][PilotPOS] * (0.5/(PilotPOS.astype(float)*(rate/Acquibuff.size))))  #Calc time error
	deltaS = deltaT*rate													  #Calc sample error

	deltaTmax = deltaT[0]  
							                                                           #select phase
	samplecorr = deltaTmax/(1/rate)			

									  #dude weed
	samplecorr2 = -int(samplecorr)	

	print samplecorr
	print samplecorr2
	
	# print AP_Acquibuff['PHASE'][CarrierPOS]														  #idk what im doin but it works

	# print deltaS

	PhaseFix = buff[(samplecorr2+(buff.size/3)):(samplecorr2+((buff.size/3)*2))] #Shift by complete samples

	PhaseFixFFT = calc_IQFFT(PhaseFix)


	test = np.array([0]*(buff.size/3), np.complex64) #Converts CS8 to CF
	test.real = buff[(samplecorr2+(buff.size/3)):(samplecorr2+((buff.size/3)*2))]['re'] # WHYYY?????
	test.imag = buff[(samplecorr2+(buff.size/3)):(samplecorr2+((buff.size/3)*2))]['im']

	testfft = np.fft.fft((test), (test.size*ifftshit))  # whatever lamo
	testfft_scale = np.fft.fftfreq((test.size*ifftshit), (1.0/float(rate))) #this works??????




	PhaseFixInit = return_amp_phase(testfft)

	deltaT = (PhaseFixInit['PHASE'][PilotPOS] * (0.5/(PilotPOS.astype(float)*(rate/Acquibuff.size))))  #Calc time error
	deltaTmax = deltaT[0]  

	deltaS = deltaT*rate													  #Calc sample error

	samplecorr = deltaTmax/(1/rate)			
	correctionfactor = samplecorr

	print deltaT

	#testfft[CarrierPOS] *= (np.cos(2*np.pi*testfft_scale[CarrierPOS]*((correctionfactor)*(1/rate)))+(1j*np.sin(2*np.pi*testfft_scale[CarrierPOS]*((correctionfactor)*(1/rate)))))

	testfft[CarrierPOS] *= np.exp((-1j*2*np.pi*CarrierPOS.astype(float)*correctionfactor*(1))/testfft.size)

	print np.exp((-1j*1*np.pi*CarrierPOS.astype(float)*correctionfactor*(1))/testfft.size)


	PhaseFixAP = return_amp_phase(testfft) 


	for i in range(10):

		deltaT = (PhaseFixAP['PHASE'][PilotPOS] * (0.5/(PilotPOS.astype(float)*(rate/Acquibuff.size))))  #Calc time error
		deltaTmax = deltaT[1] 
		samplecorr = deltaTmax/(1/rate)			
		correctionfactor = samplecorr
		print deltaT

		testfft[CarrierPOS] *= np.exp((-1j*2*np.pi*CarrierPOS.astype(float)*correctionfactor*(1))/testfft.size)

		PhaseFixAP = return_amp_phase(testfft) 


	PhaseFixAP2 = PhaseFixAP

	for i in range(10):

		deltaT = (PhaseFixAP2['PHASE'][PilotPOS] * (0.5/(PilotPOS.astype(float)*(rate/Acquibuff.size))))  #Calc time error
		deltaTmax = deltaT[2]  
		samplecorr = deltaTmax/(1/rate)			
		correctionfactor = samplecorr
		print deltaT

		testfft[CarrierPOS] *= np.exp((-1j*2*np.pi*CarrierPOS.astype(float)*correctionfactor*(1))/testfft.size)


		PhaseFixAP2 = return_amp_phase(testfft) 

	PhaseFixAP3 = PhaseFixAP2


	deltaT = (PhaseFixAP3['PHASE'][PilotPOS] * (0.5/(PilotPOS.astype(float)*(rate/Acquibuff.size))))  #Calc time error
	deltaTmax = deltaT[3]  
	samplecorr = deltaTmax/(1/rate)			
	correctionfactor = samplecorr
	print deltaT

	testfft[CarrierPOS] *= np.exp((-1j*2*np.pi*CarrierPOS.astype(float)*correctionfactor*(1))/testfft.size)

	PhaseFixAP4 = return_amp_phase(testfft) 


	deltaT = (PhaseFixAP4['PHASE'][PilotPOS] * (0.5/(PilotPOS*(rate/Acquibuff.size))))  #Calc time error
	print deltaT


	#mem = np.linalg.lstsq(testfft_scale[PilotPOS],PhaseFixAP4['PHASE'][PilotPOS])


	f, axarr = plt.subplots(6, sharex=True)
	axarr[0].plot(PhaseFixFFT['IQFFT_SCALE'], AP_Acquibuff['PHASE'])
	axarr[1].plot(PhaseFixFFT['IQFFT_SCALE'], PhaseFixInit['PHASE'])
	axarr[2].plot(PhaseFixFFT['IQFFT_SCALE'], PhaseFixAP['PHASE'])
	axarr[3].plot(PhaseFixFFT['IQFFT_SCALE'], PhaseFixAP2['PHASE'])
	axarr[4].plot(PhaseFixFFT['IQFFT_SCALE'], PhaseFixAP3['PHASE'])
	axarr[5].plot(PhaseFixFFT['IQFFT_SCALE'], PhaseFixAP4['PHASE'])


	return{'PhaseFix':PhaseFixAP, 'FFTScale':PhaseFixFFT['IQFFT_SCALE']}







sdr = init_sdr()
bigbuff = recieve_samples(5,sdr)
IQ = calc_IQFFT(bigbuff)
AmpPhase = return_amp_phase(IQ['IQFFT'])
test = fix_symbol(bigbuff)


#print test
#print AmpPhase
#print detect_carrier(AmpPhase['AMP'])

f, axarr = plt.subplots(4, sharex=True)

axarr[0].plot(IQ['IQFFT_SCALE'], AmpPhase['AMP'])
axarr[1].plot(IQ['IQFFT_SCALE'], AmpPhase['PHASE'])
axarr[2].plot(test['FFTScale'], test['PhaseFix']['AMP'])
axarr[3].plot(test['FFTScale'], test['PhaseFix']['PHASE'])
# axarr[1].plot(testfft_scale, testfft4.imag)


plt.show()