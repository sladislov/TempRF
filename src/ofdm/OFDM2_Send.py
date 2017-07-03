import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
import time
import math
import commpy
import scipy
from scipy import signal
import matplotlib.pyplot as plt


#HackRF DEFs
filterfreq = 160000
rate       = 20e6
freq   	   = 868e6+0
bufflen    = 131072
txBw	   = 20e6
txGain     = 61
txChan     = 0
txAnt	   ='TX/RX'

#OFDM DEFs
Ncarrier    = 1024
FFTlen      = (2**18)
Freq_offset = 100
Symbols     = 100
Pilot       = np.array([1+1j, 1-1j, -1-1j, -1+1j, 1+1j, 1-1j, -1-1j, -1+1j, 1+1j, 1-1j])

#PN Defs
pn_order      =    10                  ##Pn order, 2^n
pn_seed       = "0100010010"     ##Pn seed
pn_mask       = "0010000001"     ##Pn mask # "0010000001" 1 peak #

# Dtype needed for HackRF interfacing
dtype = np.dtype([('re', np.int8), ('im', np.int8)]) 

def init_sdr ():

	args = dict(driver="hackrf")
	sdr = SoapySDR.Device(args)

	sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)

	sdr.setBandwidth(SOAPY_SDR_TX, txChan, txBw)

	sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

	sdr.setGain(SOAPY_SDR_TX, txChan, txGain)

	sdr.setFrequency(SOAPY_SDR_TX, txChan, freq)

	return sdr


def gen_pn(pn_order, pn_seed, pn_mask, pn_length):   #PN generator wrapper
	pn_sequence   = commpy.pnsequence(pn_order, pn_seed, pn_mask, pn_length)
	return pn_sequence

def gen_trainer1(): #Generate first training symbol

	PN_1 = gen_pn(10, "0100010010", "0010000001", (2**10)) #generate PN sequence 2^10

	PN_1[PN_1 == 1] = 1*np.sqrt(2)									#Multiply by factor sqrt(2) for power raisins
	PN_1[PN_1 == 0] = -1*np.sqrt(2)									 #Change all 0's to -1's 

	OFDMFFT = np.array([0]*FFTlen, dtype=complex)

	OFDMFFT[100:1124:2] = PN_1[0:1023:2]+(PN_1[1:1024:2]*1j) #Fill all even frequencies with PN sequence

	# f, axarr = plt.subplots(6, sharex=False)
	# axarr[0].plot(range(OFDMFFT.size), OFDMFFT.real)
	# axarr[1].plot(range(OFDMFFT.size), OFDMFFT.imag)

	OFDMFFT  =  np.fft.ifft(OFDMFFT)

	return  add_cp(OFDMFFT, 2)


def gen_trainer2(): #generate Second training symbol

	PN_1 = gen_pn(11, "01010101101", "01000000001", (2**11)) #generate PN sequence 2^11
	PN_2 = gen_pn(11, "10101010010", "01000000001", (2**11)) #generate DIFFERENT PN sequence 2^11

	PN_1[PN_1 == 0] = -1*np.sqrt(2)	
	PN_1[PN_1 == 1] = 1*np.sqrt(2)											 #Change all 0's to -1's 
	PN_2[PN_2 == 0] = -1*np.sqrt(2)	
	PN_2[PN_2 == 1] = 1*np.sqrt(2)		


	OFDMFFT = np.array([0]*FFTlen, dtype=complex) 

	OFDMFFT[100:1124:2] = PN_1[0:1023:2]+(PN_1[1:1024:2]*1j) #Fill all even frequencies with PN sequence
	OFDMFFT[101:1124:2] = PN_2[0:1023:2]+(PN_2[1:1024:2]*1j) #Fill all uneven frequencies with PN sequence

	OFDMFFT  =  np.fft.ifft(OFDMFFT)

	return  add_cp(OFDMFFT, 2)

def gen_symbol(x):   #generate data symbol

	#np.save("pn.npy", x)

	x[x == 0] = -1
	x[x == 1] = 1

	OFDMFFT = np.array([0]*FFTlen, dtype=complex) 
	OFDMFFT[100:1124] = x[0:2048:2]*.5 + x[1:2048:2]*.5j

	##^^Insert data

	for i in range(9):
		OFDMFFT[(i+1)*100] = Pilot[i]

	##^^Add pilots

	OFDMFFT2 = OFDMFFT

	OFDMFFT  =  np.fft.ifft(OFDMFFT)

	# f, axarr = plt.subplots(6, sharex=True, sharey=True)
	# axarr[0].plot(range(OFDMFFT.size), OFDMFFT2.real)
	# axarr[1].plot(range(OFDMFFT.size), OFDMFFT2.imag)
	# axarr[2].plot(range(OFDMFFT.size), np.fft.fft(OFDMFFT).real)
	# axarr[3].plot(range(OFDMFFT.size), np.fft.fft(OFDMFFT).imag)

	#plt.show()

	return  add_cp(OFDMFFT, 2)



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

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def add_cp(symbol, cp):

	print "hoi"

	symbol3 = symbol[(symbol.size-(symbol.size/cp)):symbol.size]

	print symbol3.size

	symbol2 = np.append(symbol3, symbol)

	print symbol2.size

	return symbol2





symbol1 = gen_trainer1()

symbol2 = gen_trainer2()

empty = np.array([0]*(symbol1.size), dtype=complex)

symbol4 = gen_symbol(gen_pn(11, "01010101101", "01000000001", (2**11))) # 11, "01010101101", "01000000001", (2**11))

symbol5 = gen_symbol(gen_pn(11, "00101011000", "01000000001", (2**11))) # 11, "01010101101", "01000000001", (2**11))

a =  np.asarray(tobits("Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. LWJLWJLW")) # 11, "01010101101", "01000000001", (2**11))

b =  np.asarray(tobits("Hoi, Attila zijn moeder is echt heel geil, en ehh zijn zusje ook. Ik zou ze allemaal doen echt waar zeker weten wel hoiiii. Ik vreet nu kapsalon echt k$$$$$ dik joeeeee de koelkast is heel vies die hebben we schoongemaakt maar toen werd bootmongool boos :)"))



# symbol4 = gen_symbol(a)

# symbol5 = gen_symbol(b)
					
biggobuffer = np.append(symbol1, [symbol2, symbol4, symbol5, symbol4, symbol5, symbol4, symbol5, symbol4])

biggobuffer_CS = cf_to_cs(biggobuffer)


#symbol4fft = np.fft.fft(cf_to_cs(symbol4[131072:131072*3]))



# f, axarr = plt.subplots(6, sharex=False)
# axarr[0].plot(range(symbol4fft.size), symbol4fft.imag)
# axarr[1].plot(range(symbol2.size), symbol2.real)
# axarr[2].plot(range(biggobuffer.size), biggobuffer)
# axarr[3].plot(range(biggobuffer.size), biggobuffer_CS['re'])
# axarr[4].plot(range(biggobuffer.size), biggobuffer_CS['im'])

#plt.show()



sdr = init_sdr()
txStream = sdr.setupStream(SOAPY_SDR_TX, "CS8", [txChan])
print("Activating Tx stream")
sdr.activateStream(txStream)

print biggobuffer_CS

print float(biggobuffer.size)/131072.0

while True:
	for i in range (biggobuffer_CS.size/131072):
		a = i*131072
		b = ((i*131072)+131072)
		sr = sdr.writeStream(txStream, (biggobuffer_CS[a:b]['re'],biggobuffer_CS[a:b]['im']), biggobuffer_CS.size)
		#print sr
		
		#print sr
		#print i
		#print biggobuffer_CS[a:b].size
