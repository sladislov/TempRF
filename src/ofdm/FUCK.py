import serial
import time



ser = serial.Serial('/dev/ttyUSB1')
ser.baudrate = 115200


while True:
    ser.write(b'Whether it is Snapchat, Twitter, Facebook, Yelp or just a note to co-workers or business officials, the number of actual characters matters. What you say may not be as importantaa\n')
    time.sleep(.1)