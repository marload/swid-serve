import os

basePATH = "./dataset"


for i in range(1, 26):
    modelPATH = os.path.join(basePATH, "model{}".format(i))
    for j in range(2):
        if j == 0:
            splitPATH = os.path.join(modelPATH, 'train')
        else:
            splitPATH = os.path.join(modelPATH, 'test')
        
        for k in range(2):
            if k == 0:
                finalPATH = os.path.join(splitPATH, "crack")
            else:
                finalPATH = os.path.join(splitPATH, "normal")
            imageList = os.listdir(finalPATH)
            for img in imageList:
                if "aug_" in img:
                    os.remove(os.path.join(finalPATH, img))
                    print("{} is deleted.".format(img))
