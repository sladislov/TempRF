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
import serial

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

# Dtype needed for HackRF interfacing
dtype = np.dtype([('re', np.int8), ('im', np.int8)]) 

#open serial
ser = serial.Serial('/dev/ttyUSB0')
ser.baudrate = 115200
print ser


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

	#print "hoi"

	symbol3 = symbol[(symbol.size-(symbol.size/cp)):symbol.size]

	#print symbol3.size

	symbol2 = np.append(symbol3, symbol)

	#print symbol2.size

	return symbol2

def get_usart_data():

	rec = ser.read(400)
	a = rec.find('\n')
	b = rec.find('\n', a+1)
	print (a, b)
	data = rec[a:b]
	if (a && b) == 0:
		data = rec[0:1440]

	
	print data
	print len(data)

	return tobits(data)

def encode(data):
	enc_data = np.array([0]*(2048*7))

	for i in range(9):
		enc_data[i*1440:(i+1)*1440] = data

	return enc_data

def create_signal(data, trainer1, trainer2, empty):



	# symbol4 = gen_symbol(gen_pn(11, "01010101101", "01000000001", (2**11))) # 11, "01010101101", "01000000001", (2**11))

	# symbol5 = gen_symbol(gen_pn(11, "00101011000", "01000000001", (2**11))) # 11, "01010101101", "01000000001", (2**11))

	# biggobuffer = np.append(trainer1, [trainer2, symbol4, symbol5, symbol4, symbol5, symbol4, symbol5, symbol4])
#############^^^^^^^^^^ TEST SIGNAL

	symbol1 = gen_symbol(data[0    : 2048])
	symbol2 = gen_symbol(data[2048 : 4096])
	symbol3 = gen_symbol(data[4096 : 6144])
	symbol4 = gen_symbol(data[6144 : 8192])
	symbol5 = gen_symbol(data[8192 :10240])
	symbol6 = gen_symbol(data[10240:12288])
	symbol7 = gen_symbol(data[12288:14336])
	print symbol1.size

	biggobuffer = np.append(trainer1, [trainer2, symbol1, symbol2, symbol3, symbol4, symbol5, symbol6, symbol7])
	biggobuffer_CS = cf_to_cs(biggobuffer)

	return biggobuffer_CS

def main():

	bigbuff = np.array([0]*(393216*9))

	trainer1 = gen_trainer1()
	trainer2 = gen_trainer2()
	empty = np.array([0]*(trainer1.size), dtype=complex)

	data = np.array([0]*1440)

	enc_data = np.array([0]*(2048*7))

	if os.path.isfile('lock.npy'): #Prevent lockouts
		os.remove('lock.npy')

	while True:


		data = get_usart_data()

		enc_data = encode(data)

		bigbuff = create_signal(enc_data, trainer1, trainer2, empty)

		if os.path.isfile('lock.npy'):
			print "LOCKED"
		else:
			np.save("lock", 0)
			np.save("bigbuff", bigbuff)
			os.remove('lock.npy')

		time.sleep(0.5)
		print "Joe"

main()