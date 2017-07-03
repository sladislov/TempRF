#Recieve samples and do all the TIME DOMAIN reciever shenanigans 



import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
import scipy
import commpy
from scipy import signal
import math
from mpl_toolkits.mplot3d import Axes3D
import time
import Rectest
import os
from subprocess import call


dtype = np.dtype([('re', np.int8), ('im', np.int8)]) # MAGIC DONT TOUCH THIS FUCK

filterfreq = 160000
rate       = 20e6
frequency  = 868e6+0
length     = 0e0
amplif     = 30
bufflen    = 131072
bandwidth  = .5e6
FFTlen      = (2**18)
x = 0

half = (131072)
full = (262144)
length = 4e5*9

test_list = ['test', 'list', half]

## FOR V4

def init_sdr ():

	#enumerate devices
	results = SoapySDR.Device.enumerate()
	#for result in results: print(result)

	#Create HackRF instance
	args = dict(driver="hackrf")
	sdr = SoapySDR.Device(args)

	#Apply settings

	sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
	sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
	sdr.setGain(SOAPY_SDR_RX, 0, amplif)
	sdr.setBandwidth(SOAPY_SDR_RX, 0, bandwidth)
	rxStream = sdr.setupStream(SOAPY_SDR_RX, "CS8")
	sdr.activateStream(rxStream) #start streaming

	

	return {'sdr':sdr, 'rxStream':rxStream}


def recieve_samples(cycle, sdr, rxStream):
	#create a re-usable buffer for rx
	buff = np.array([0]*bufflen, np.complex64)
	testbuff = np.zeros(bufflen, dtype)  #FUCKING MAGICAL 
	bigbuff = np.array([0]*0, dtype)

	#setup a stream (complex integer)
	

	#receive some samples
	for i in range(cycle):
	    sr = sdr.readStream(rxStream, [testbuff], len(testbuff), timeoutUs=1000000)
	    #print(sr.ret) #num samples or error code
	    #print(testbuff)
	    if (i > 1):
	        bigbuff = np.append(bigbuff, testbuff)
	return bigbuff

def cs_to_cf(cs): #????

	CF = np.array([0]*(cs.size), np.complex64) #Converts CS8 to CF
	CF.real = bigbuff[0:cs.size]['re']
	CF.imag = bigbuff[0:cs.size]['im']

	return CF

def gen_pn(pn_order, pn_seed, pn_mask, pn_length):   #PN generator wrapper
	pn_sequence   = commpy.pnsequence(pn_order, pn_seed, pn_mask, pn_length)
	return pn_sequence



def detect_start(buffer):

	return 1


def fix_frequency(buffer, start):

	return 1

def decode_fft():
	PN_1 = gen_pn(11, "01010101101", "01000000001", (2**11)) #generate PN sequence 2^11
	PN_2 = gen_pn(11, "10101010010", "01000000001", (2**11)) #generate DIFFERENT PN sequence 2^11

	PN_1[PN_1 == 0] = -1									 #Change all 0's to -1's 
	PN_2[PN_2 == 0] = -1


	OFDMFFT = np.array([0]*FFTlen, dtype=complex) 

	OFDMFFT[100:1124:2] = PN_1[0:1023:2]+(PN_1[1:1024:2]*1j) #Fill all even frequencies with PN sequence
	OFDMFFT[101:1124:2] = PN_2[0:1023:2]+(PN_2[1:1024:2]*1j) #Fill all uneven frequencies with PN sequence

	return OFDMFFT

def decode_fft2():   #generate data symbol

	x = gen_pn(12, "010101001101", "100101000001", (2**12))

	#np.save("pn.npy", x)

	x[x == 0] = -1

	OFDMFFT = np.array([0]*FFTlen, dtype=complex) 
	OFDMFFT[100:1124] = x[0:2048:2] + x[1:2048:2]*1j

	return  OFDMFFT


def schmidl_filter(bigbuff_CF):
	x = 0.0+0.0j
	y = np.array([0]*(bigbuff_CF.size-2*half), np.complex64)

	bigbuff_CF_CONJ = np.conj(bigbuff_CF)

	# for i in range(bigbuff_CF.size-2*half):
	# 	x += ((bigbuff_CF_CONJ[i+half])*bigbuff_CF[i+(2*half)]) - (bigbuff_CF_CONJ[i]*bigbuff_CF[i+half])
	# 	y[i] = (x)

	# 	#if x > 8e5:
	# 		#break
	# 	if (i%1000000 == 0):
	# 		print i

	# f, axarr = plt.subplots(2, sharex=True)
	# axarr[0].plot(range(y.size), y.real)

	for i in range((bigbuff_CF.size-2*half)/100):
		x += ((bigbuff_CF_CONJ[(i*100)+half])*bigbuff_CF[(i*100)+(2*half)]) - (bigbuff_CF_CONJ[i*100]*bigbuff_CF[i*100+half])
		y[i*100:(i+1)*100] = (x)

	orisch = np.copy(y)

	if y.max() < 10000:
		return 0

	y[16000000:bigbuff_CF.size] = 0
	y[y > y.max()*0.9999] = 1e10

	y = np.diff(y)

	# f, axarr = plt.subplots(2, sharex=True)
	# axarr[0].plot(range(y.size), y.real)
	# plt.show()

	#print 'waarom'

	########################^^^^ Detect schmidl peaks

	a = np.where(y > 100000)[0]

	b = np.where(y < -100000)[0]

	c = np.array([])

	for i in range(a.size-1):
		if abs(a[i] - a[i+1]) < 1e6:
		  c = np.append(c, i+1)
                    
	a = np.delete(a, c)

	c = np.array([])

	for i in range(b.size-1):
		if abs(b[i] - b[i+1]) < 1e6:
			c = np.append(c, i+1)
	b = np.delete(b, c)

	#######################^^^^^^^ sanity check schmidl


	#afull = int(((a[1]-a[0])/13.5))


	#afull = full + 0

	#half = afull/2

	z = np.sum(y == 1e10)

	offset = half + full

	start = int(b[0]+a[0])/2
	#print orisch[start]
	phase = np.angle(orisch[start])

	#print phase

	deltaF = phase/(np.pi*(262144.0/20000000.0))
	print deltaF
	y = np.zeros(y.size)

	y[a] = 1e10
	y[b] = -1e10
	y[start] = 7e9
	y[start+offset] = 5e9
	y[start+offset+full] = 5e9
	y[start+offset+offset+offset] = 5e9
	y[start+offset+offset+offset+full] = 5e9

	return {'SchmidlPeaks':y, 'start':start ,'offset':offset, 'full':full, 'half':half, 'phase':phase, 'orisch':orisch}
	#############################  Symbol Synch ^^^^^^^^^^^^^

def fix_freq(bigbuff, phase):
	a = -2.0j*np.pi*phase

	for i in range((bigbuff.size/1000)-1000):
		bigbuff[(i*1000):((i+1)*1000)] *= np.exp((a*(float(i+131072))/float(full+half)))
		# if (i%100000000 == 0):
		# 	print i 
	return bigbuff
	#np.save ("testdata", bigbuff)


sdrandstrm = init_sdr()

sdr = sdrandstrm['sdr']
rxStream = sdrandstrm['rxStream']


plt.ion()
f, axarr = plt.subplots(2, sharex=True)
start_time    = time.time()

while True:



	print test_list
	print "recieving"

	bigbuff = recieve_samples(153, sdr, rxStream)

	bigbuff_CF = cs_to_cf(bigbuff)

	print "schmidlling"
	schmidl = schmidl_filter(bigbuff_CF)

	if schmidl != 0:


		print "fixing frequency"
		bigbuff_CF = fix_freq(bigbuff_CF[schmidl['start']:schmidl['start']+length],schmidl['phase'])
		print "plotting"

		# plt.clf()
		# plt.plot(range(bigbuff_CF.size/100), (bigbuff_CF.real[::100]))
		# #plt.plot(range(schmidl['orisch'].size), schmidl['orisch'])
		# plt.pause(0.05)
		os.system('clear')
		print schmidl['phase']
		print int(time.time())
		if (np.sum(np.abs(bigbuff_CF.real))/bigbuff_CF.size)>15:
			amplif -= 2
		if (np.sum(np.abs(bigbuff_CF.real))/bigbuff_CF.size)<15:
			amplif += 2	

		if amplif > 116:
			amplif = 116
		sdr.setGain(SOAPY_SDR_RX, 0, amplif)

		print"decoding"
		Rectest.main(bigbuff_CF, schmidl)
		#call('Rectest.py')
	else:
		os.system('clear')
		print "NO HIT"
	



############## Experiments below

