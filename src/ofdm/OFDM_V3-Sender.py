import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
import time
import math
import commpy
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from multiprocessing import Process
import os

#HackRF DEFs
filterfreq = 160000
rate       = 20e6
freq   	   = 868e6+0
bufflen    = 131072
txBw	   = 20e6
txGain     = 61
txChan     = 0
txAnt	   ='TX/RX'

def init_sdr ():

	args = dict(driver="hackrf")
	sdr = SoapySDR.Device(args)

	sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)

	sdr.setBandwidth(SOAPY_SDR_TX, txChan, txBw)

	sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

	sdr.setGain(SOAPY_SDR_TX, txChan, txGain)

	sdr.setFrequency(SOAPY_SDR_TX, txChan, freq)

	return sdr



def main():
	sdr = init_sdr()
	txStream = sdr.setupStream(SOAPY_SDR_TX, "CS8", [txChan])
	print("Activating Tx stream")
	sdr.activateStream(txStream)

	if os.path.isfile('lock.npy'): #Prevent lockouts
		os.remove('lock.npy')

	while True:

		if os.path.isfile('lock.npy'):
			print "LOCKED"
		else:
			np.save("lock", 0)
			try:
				biggobuffer_CS = np.load("bigbuff.npy")
			except:
				"error JOE"

			os.remove('lock.npy')

		time.sleep(.7)

		for i in range(3):
			print i
			for i in range (biggobuffer_CS.size/131072):
				a = i*131072
				b = ((i*131072)+131072)
				sr = sdr.writeStream(txStream, (biggobuffer_CS[a:b]['re'],biggobuffer_CS[a:b]['im']), biggobuffer_CS.size)
				#print sr

main()