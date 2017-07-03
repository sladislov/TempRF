import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np #use numpy for buffers
import matplotlib.pyplot as plt
import scipy
import commpy
from scipy import signal
import math
from mpl_toolkits.mplot3d import Axes3D
import sys

rate       = 20e6
FFTlen      = (2**18)
x = 0

half = (131072)
full = (262144)
offset = half+full
Pilot       = np.array([1+1j, 1-1j, -1-1j, -1+1j, 1+1j, 1-1j, -1-1j, -1+1j, 1+1j, 1-1j])


def gen_pn(pn_order, pn_seed, pn_mask, pn_length):   #PN generator wrapper
	pn_sequence   = commpy.pnsequence(pn_order, pn_seed, pn_mask, pn_length)
	return pn_sequence

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

def cs_to_cf(cs): #????

	CF = np.array([0]*(cs.size), np.complex64) #Converts CS8 to CF
	CF.real = bigbuff[0:cs.size]['re']
	CF.imag = bigbuff[0:cs.size]['im']

	return CF

def frombits(bits):
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def P2R(radii, angles):
    return radii * np.exp(1.0j*angles)

def decode_constell(symbol):

	symbol_decode = np.array([0]*2048, int)
	for i in range(1024):
		if abs(symbol.real[i+100]) > abs(symbol.imag[i+100]):
			#symbol3.real[i+100] *= 10
			if symbol.real[i+100] > 0:
				symbol_decode[(i*2)]   = 1
				symbol_decode[(i*2)+1] = 1
			else:
				symbol_decode[(i*2)]   = 0
				symbol_decode[(i*2)+1] = 0
		else:
			#symbol3.imag[i+100] *= 10
			if symbol.imag[i+100] > 0:
				symbol_decode[(i*2)]   = 0
				symbol_decode[(i*2)+1] = 1
			else:
				symbol_decode[(i*2)]   = 1
				symbol_decode[(i*2)+1] = 0

	return symbol_decode
def decode_bits(symbol):
	result = np.array([0]*1440, long)
	for i in range(7):
		result += symbol[0+(i*1440):1440+(i*1440)]
	result = np.round(result.astype(float) / 7.0)



	return result
def main(bigbuff, schmidl):
	# bigbuff = np.load('IQ.npy')
	# schmidl = np.load('SCH.npy')

	#print schmidl

	symbol1 = np.fft.fft(bigbuff[offset*1:offset*1+full])
	symbol2 = np.fft.fft(bigbuff[offset*2:offset*2+full])
	symbol3 = np.fft.fft(bigbuff[offset*3:offset*3+full])
	symbol4 = np.fft.fft(bigbuff[offset*4:offset*4+full])
	symbol5 = np.fft.fft(bigbuff[offset*5:offset*5+full])
	symbol6 = np.fft.fft(bigbuff[offset*6:offset*6+full])
	symbol7 = np.fft.fft(bigbuff[offset*7:offset*7+full])
	symbol8 = np.fft.fft(bigbuff[offset*8:offset*8+full])

	#print offset*8
	#print offset*8+full


	#################

	a = symbol1[100:1124]

	b = decode_fft()[100:1124]

	a = np.angle(a)

	b = np.angle(b)

	decode = a-b
	##print decode
	decode = P2R(1.0, decode)

	####################^^Get initial decoding
	symbol1[100:1124] *= decode[::-1]
	symbol2[100:1124] *= decode[::-1]
	symbol3[100:1124] *= decode[::-1]
	symbol4[100:1124] *= decode[::-1]
	symbol5[100:1124] *= decode[::-1]
	symbol6[100:1124] *= decode[::-1]
	symbol7[100:1124] *= decode[::-1]
	symbol8[100:1124] *= decode[::-1]
	#symbol3[100:1124] *= decode[::-1]


	##print np.angle(Pilot)
	##print np.angle(symbol1[100:1024:100])
	##print (np.angle(Pilot)-np.angle(symbol1[100:1024:100]))

	##print "ea"
	symbol3_scale = np.fft.fftfreq((symbol3.size), (1.0/float(rate)))


	symbol1 = symbol1 * P2R(1, (-np.angle(symbol1[500])))
	symbol2 = symbol2 * P2R(1, (-np.angle(symbol2[500])))
	symbol3 = symbol3 * P2R(1, (-np.angle(symbol3[500])))
	symbol4 = symbol4 * P2R(1, (-np.angle(symbol4[500])))
	symbol5 = symbol5 * P2R(1, (-np.angle(symbol5[500])))
	symbol6 = symbol6 * P2R(1, (-np.angle(symbol6[500])))
	symbol7 = symbol7 * P2R(1, (-np.angle(symbol7[500])))
	symbol8 = symbol8 * P2R(1, (-np.angle(symbol8[500])))

	# symbol3.real = symbol3.real /(symbol3.real.max())
	# symbol3.imag = symbol3.imag /(symbol3.imag.max())


	############################ Make constellation ^^^^^^^^^

	symbol1_decode = decode_constell(symbol1)
	symbol2_decode = decode_constell(symbol2)
	symbol3_decode = decode_constell(symbol3)
	symbol4_decode = decode_constell(symbol4)
	symbol5_decode = decode_constell(symbol5)
	symbol6_decode = decode_constell(symbol6)
	symbol7_decode = decode_constell(symbol7)
	symbol8_decode = decode_constell(symbol8)

	###^^ Decode

	a = gen_pn(11, "01010101101", "01000000001", (2**11))

	b = gen_pn(11, "00101011000", "01000000001", (2**11))

	c = decode_constell(decode_fft())

	#a =  tobits("Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. Dus ik zeg ze tegen die bitch. LWJLWJLW") # 11, "01010101101", "01000000001", (2**11))
	 
	a = np.asarray(a)

	print int(((sum(symbol1_decode == c)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol2_decode == a)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol3_decode == b)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol4_decode == a)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol5_decode == b)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol6_decode == a)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol7_decode == b)-1024.0)/1024.0)*100.0)
	print int(((sum(symbol8_decode == a)-1024.0)/1024.0)*100.0)

	biggosymbol = np.append(symbol2_decode, [symbol3_decode, symbol4_decode, symbol5_decode, symbol6_decode, symbol7_decode, symbol8_decode])
	print decode_bits(biggosymbol)
	print frombits(biggosymbol)



	# print frombits(symbol2_decode)
	# print frombits(symbol3_decode)
	# print frombits(symbol4_decode)
	# print frombits(symbol5_decode)
	# print frombits(symbol6_decode)
	# print frombits(symbol7_decode)
	# print frombits(symbol8_decode)

	###^ Display values



	# symbol1_decode = 1-(a[0:2048] == symbol1_decode)

	plt.ion()
	plt.clf()
	plt.scatter(symbol2.real[100:1123], symbol2.imag[100:1123])
	plt.pause(0.01)




	# f, axarr = plt.subplots(7, sharex=True, sharey=True)
	# axarr[0].set_title('constellage 2')
	# axarr[0].scatter(symbol2.real[100:1123], symbol2.imag[100:1123])
	# axarr[1].set_title('constellage 3')
	# axarr[1].scatter(symbol3.real[100:1123], symbol3.imag[100:1123])
	# axarr[2].set_title('constellage 4')
	# axarr[2].scatter(symbol4.real[100:1123], symbol4.imag[100:1123])
	# axarr[3].set_title('constellage 5')
	# axarr[3].scatter(symbol5.real[100:1123], symbol5.imag[100:1123])
	# axarr[4].set_title('constellage 6')
	# axarr[4].scatter(symbol6.real[100:1123], symbol6.imag[100:1123])
	# axarr[5].set_title('constellage 7')
	# axarr[5].scatter(symbol7.real[100:1123], symbol7.imag[100:1123])
	# axarr[6].set_title('constellage 8')
	# axarr[6].scatter(symbol8.real[100:1123], symbol8.imag[100:1123])

	# f, axarr = plt.subplots(6, sharex=True)
	# axarr[0].plot(range(bigbuff.size), bigbuff.real)
	# axarr[0].set_title('I')
	# axarr[1].plot(range(bigbuff.size), bigbuff.imag)
	# axarr[1].set_title('Q')
	# axarr[4].plot(range(decode.size), decode.real)




	# f = decode_fft()

	# test = symbol1 #* (f.real+f.imag[::-1])

	# f = decode_fft2()

	# test2 = symbol3#* (f.real+f.imag[::-1])

	# meme = test+test2


	# fig = plt.figure()
	# ax = fig.add_subplot(311, projection='3d')

	# ax.scatter(test.real[99:1124], test.imag[99:1124], range(1025))
	# ax.set_title('1 phasefault')

	# ax = fig.add_subplot(312, projection='3d')
	# ax.scatter(test2.real[99:1124], test2.imag[99:1124], range(1025))
	# ax.set_title('2 phasefault')


	# ax = fig.add_subplot(313, projection='3d')

	# ax.scatter(meme.real[99:1124], meme.imag[99:1124], range(1025))
	# ax.set_title('1-2 phasefault')
	#plt.show()