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

OFDMsample = np.load('OFDM.npy')

print OFDMsample.size

while True:

    
    sr = sdr.writeStream(txStream, [OFDMsample], (OFDMsample.size), timeoutUs=1000000)
    print sr







