import extcolors
import argparse
import cv2
import csv
import sys
from tqdm import tqdm
import os
TRAIN_DIR = 'images/'

k = 0
training_data = []
path = []
flist = []
flist1 = []
i = 0
for img in tqdm(os.listdir(TRAIN_DIR)):

    f = open('csvall.csv', 'rU')  # open the file in read universal mode
    output = []
    output1 = []
    output2 = []
    for line in f:

        cells = line.split(",")
        output.append((cells[1]))
        output1.append((cells[7]))
        output2.append((cells[4]))

        # since we want the first, second and third column

    j = 0

    for i in output:
        d = str(output[j])
        e = str(output1[j])
        ff = str(output2[j])
        j = j + 1

        if (img == d and e == "1"):
            flist.append(d)
            flist1.append(ff)

            k = k + 1
    f.close()
print(flist)
print(len(flist))
print(flist1)
# np.save('train_data.npy', training_data)
# return training_data

j = 0
for i in range(0, 80):
    d = str(flist[j])
    e = "/home/ananth/images/"+d
    print(e)
    img = cv2.imread(e, 1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    icrop = img[0:0+1000, 0:0+1000]
    cv2.imwrite("r.jpg", icrop)

    colors, pixel_count = extcolors.extract("r.jpg")
    print (colors)
    j=j+1
    '''
    row = [colors[0][0][0], colors[0][0][1], colors[0][0][2], colors[0][1], colors[1][0][0], colors[1][0][1],
           colors[1][0][2], colors[1][1], colors[2][0][0], colors[2][0][1], colors[2][0][2], colors[2][1],colors[3][0][0], colors[3][0][1], colors[3][0][2], colors[3][1],
           colors[4][0][0], colors[4][0][1], colors[4][0][2], colors[4][1],colors[5][0][0], colors[5][0][1], colors[5][0][2], colors[5][1],
           flist[j],flist1[j]]
    j=j+1;
    with open('ripeness.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        j = j+1
        csvFile.close()
'''