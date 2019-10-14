import extcolors
import argparse
import cv2
import csv
import sys
output=[]
f =open('csvtry.csv','rU')
for line in f:
	
	cells = line.split(",")
	output.append(cells[0])
f.close()
print (output)