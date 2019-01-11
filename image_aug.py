from imgaug import augmenters as iaa
import cv2
import os

PATH = []
for i in range(1, 26):
    PATH.append("./dataset/model{}/train/normal".format(i))

def generator(image_list):

    for name in image_list:
        fileName = name
        name = os.path.join(PATH, name)
        images = cv2.imread(name)
        
        
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        seq = iaa.Sequential([
            iaa.Flipud(p=0.5),
            iaa.Fliplr(p=0.5),
            sometimes(iaa.Pepper(p=0.10)),
            sometimes(iaa.Salt(p=0.03)),
            sometimes(iaa.AdditivePoissonNoise(lam=8.0)),
            sometimes(iaa.JpegCompression(compression=50)),
            sometimes(iaa.PiecewiseAffine(scale=0.015)),
            sometimes(iaa.MotionBlur(k=7, angle=0)),
            sometimes(iaa.MotionBlur(k=5, angle=144))
        ], random_order=False)

        for i in range(10):

            images_aug = seq.augment_image(images)
            name = 'aug_' + fileName.split('.')[0] + "-" + str(i)+'.jpg'
            name = os.path.join(PATH, name)
            cv2.imwrite(name, images_aug)
            print(name + " is saved.")
        

for i in PATH:
    generator(os.listdir(i))
