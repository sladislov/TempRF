########################################################################
## Will disrupt RF for cash
########################################################################
## Do not comment about format you will trigger me
########################################################################
##
## SDR_TX_PN
## Test script for HackRF PN sounding transmitter
## Copyright Laszlo Olk 2018-1
##
########################################################################
## THIS NEEDS SDR_RX_PN TO GENERATE A PN SEQUENCE TO SEND
## IT WILL NOT WORK WITHOUT
## SWEAR TO GOD IF YOU MAIL ME I WILL BITE YOUR TORSO AND GIVE YOU A DISEASE
########################################################################
import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
import time
import math
import commpy
import matplotlib.pyplot as plt

ampl=0.0
filterfreq=800000
freq=868e6
rate=1e6
txBw=20e6
txChan=0
rxChan=0
txGain=61
txAnt='TX/RX'
clockRate=10e6
waveFreq=rate/10
dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK


args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)


#set sample rate
sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
print("Actual Tx Rate %f Msps"%(sdr.getSampleRate(SOAPY_SDR_TX, txChan)/1e6))

#set bandwidth
sdr.setBandwidth(SOAPY_SDR_TX, txChan, txBw)

#set antenna
print("Set the antenna")
sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

#set overall gain
print("Set the gain")
sdr.setGain(SOAPY_SDR_TX, txChan, txGain)


#tune frontends
print("Tune the frontend")
sdr.setFrequency(SOAPY_SDR_TX, txChan, freq)

#list gain
print("\nCurrent Gain:")
print(sdr.getGain(SOAPY_SDR_TX, 0))
#list freq
print("\nCurrent Frequency:")
print(sdr.getFrequency(SOAPY_SDR_TX, 0))


#create tx stream
#creating stream
print("Creating Tx stream")
txStream = sdr.setupStream(SOAPY_SDR_TX, "CS8", [txChan])
print("Activating Tx stream")
sdr.activateStream(txStream)


#load samples



pn_sample = np.load("PN_filter.npy")

#sampsCh0 = np.load("dikkememes.npy")

#print pn_sample.size()/streamMTU 

#puturshithere#

#get MTU
streamMTU = sdr.getStreamMTU(txStream)
print 'Maximum transfer is'
print streamMTU
print 'Setting sample size to 65,535 for prime reasons'
streamMTU = 131070
print 'Current transfer is'
print streamMTU

#create sample channel make it bigly as a joke ( get it to 1.048 seconds do the math lazy fuck)

sampsCh0 = np.zeros(streamMTU, dtype)




#list variables
timeLastPrint = time.time()
totalSamps = 0
mem = 0
i = 0
j = 0
memes = 0.0


#####
##### MOVED ALL PN SEQUENCE GENERATION AND FILTERING TO OTHER SCRIPT
##### AS A JOKE OF COURSE
#####

#create pn sequence
#pnseq = commpy.pnsequence (16, "1010101010101010", "1010101010101010", 65535)



###Interpoleren m.b.t. filter
###1 sec Pn Lengte
###FFTW (indien niet in slang)
###PN sequence bandbreedte -> reflecties?
###Nul opvulling

print sampsCh0
print sampsCh0.size
sampsCh1 = np.array([0]*112346, np.complex64)
while True:

    
    sampsCh0 = np.copy(pn_sample[(0+112347*j):(112346+112347*j)])
    sampsCh1.real = sampsCh0['re']
    sampsCh1.imag = sampsCh0['im']
    
    sr = sdr.writeStream(txStream, [sampsCh0['re']], (sampsCh0.size), timeoutUs=1000000)


    #Display signal error if too many loads
    if sr.ret != sampsCh0.size:
        raise Exception("Expected writeStream() to consume all samples! %d"%sr.ret)
    totalSamps += sr.ret

    
    #Display signal generation rate
  #  if time.time() > timeLastPrint + 5.0:
  #      print("Python siggen rate: %f Msps"%(totalSamps/(time.time()-timeLastPrint)/1e6))
  #      totalSamps = 0
  #      timeLastPrint = time.time()


    j = j + 1
    if j == 7:
        j = 0
    #print j



print("Deactivating and closing stream")
sdr.deactivateStream(txStream)
sdr.closeStream(txStream)
print("Dont get caught by the FCC")
