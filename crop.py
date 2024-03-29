import csv
import random
import math
import extcolors
import argparse
import cv2
import sys
from tqdm import tqdm
import os
import operator
'''
with open('/home/sheetal/Desktop/ExpertsOpinion_15GRADES_SIH_2019_TRAIN.xlsx','rt') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print (', '.join(row))
'''       
def loadDataSet(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(1,len(dataset)-1):
            for y in range(16):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

trainingSet=[]
testSet=[]
loadDataSet(r'col1.csv',0.66,trainingSet, testSet)
print ('Train: ' + repr(len(trainingSet)))
print ('Test: ' + repr(len(testSet)))


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)
'''
data1= [2, 2, 2, 'a']
data2= [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print 'Distance:' +repr(distance)
'''

def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors






def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if(response in classVotes):
            classVotes[response] +=1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]       
   

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


prediction=[]
ex=[]
accuracy=[]
neighbors=[]
len1=len(testSet)
for i in range(1,len1):
	neighbors= getNeighbors(trainingSet, testSet[i],3)
	response=getResponse(neighbors)
	prediction.append(response)
	ex.append(testSet[i][-1])
print (prediction)
print(ex)
accuracy = getAccuracy(ex, prediction)
print(accuracy)
