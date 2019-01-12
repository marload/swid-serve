import os
import shutil
import random
import argparse


def filesListInTheDirectory(PATH, shuffle=True):
    filesList = os.listdir(PATH)
    if shuffle:
        random.shuffle(filesList)
    return filesList


def directoryLength(PATH):
    return len(os.listdir(PATH))

def numOfImageToMove(PATH, splitRate):
    return splitRate

parser = argparse.ArgumentParser()
parser.add_argument('--model_index', type=int, default=5)
parser.add_argument('--crackSplitRate', type=int, default=100)
parser.add_argument('--normalSplitRate', type=int, default=100)

config = parser.parse_args()

modelIdx = config.model_index
crackSplitRate = config.crackSplitRate
normalSplitRate = config.normalSplitRate
PATH = "./dataset/model{}".format(modelIdx)

sourceBase = os.path.join(PATH, "train")
destinationBase = os.path.join(PATH, "test")

sourceCrack = os.path.join(sourceBase, "crack")
sourceNormal = os.path.join(sourceBase, "normal")

destinationCrack = os.path.join(destinationBase, "crack")
destinationNormal = os.path.join(destinationBase, "normal")


numOfCrackDataToMove = numOfImageToMove(sourceCrack, crackSplitRate)
numOfNormalDataToMove = numOfImageToMove(sourceNormal, normalSplitRate)

print("Start Move CracK Data")
crackDataList = filesListInTheDirectory(sourceCrack, shuffle=True)
for i in range(numOfCrackDataToMove):
    fileName = crackDataList.pop()
    sourcePath = os.path.join(sourceCrack, fileName)
    destinationPath = os.path.join(destinationCrack, fileName)
    shutil.move(sourcePath, destinationPath)
    print("\t[{}] -> [{}]".format(sourcePath, destinationPath))
print("End Move Crack Data")
print()

print("Start Move Normal Data")
normalDataList = filesListInTheDirectory(sourceNormal, shuffle=True)
for i in range(numOfNormalDataToMove):
    fileName = normalDataList.pop()
    sourcePath = os.path.join(sourceNormal, fileName)
    destinationPath = os.path.join(destinationNormal, fileName)
    shutil.move(sourcePath, destinationPath)
    print("\t[{}] -> [{}]".format(sourcePath, destinationPath))
print("End Move Normal Data")


