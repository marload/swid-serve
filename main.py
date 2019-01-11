import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)

config = parser.parse_args()

normalRecallDict = {}
crackRecallDict = {}
accuracyDict = {}

for indexModel in range(1, 26):
    print('===> Loading Data...')
    modelName = indexModel
    epochs = config.epochs
    batch_size = config.batch_size
    num_workers = config.num_workers

    transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    basePATH = "./dataset"
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(basePATH, "model{}/train".format(modelName)),
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.ImageFolder(root=os.path.join(basePATH, "model{}/test".format(modelName)),
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('crack', 'normal')

    print('===> Building Model - ResNet18 - ...')

    net = models.inception_v3()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("\n===> Training Start")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if torch.cuda.device_count() > 1:
        print("\n===> Training on GPU")
        net = nn.DataParallel(net)



    for epoch in range(epochs):
        print('\n===> epoch {}'.format(epoch+1))
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 0:
                print("[{}, {}] loss: {:.4f}".format(epoch+1, i, running_loss / 30))
                running_loss = 0.0

    print("\n===> Finished Training...")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("\nAccuracy of the network on the {} test images: {}".format(len(testloader)*batch_size, 100 * correct / total))
    accuracyDict[indexModel] = 100 * correct / total

    class_correct = [0.0, 0.0]
    class_total = [0.0, 0.0]

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print("\nAccuracy of {} : {}".format(classes[i], 100*class_correct[i] / class_total[i]))
        if i == 0:
            crackRecallDict[indexModel] = 100*class_correct[i] / class_total[i]
        if i == 1:
            normalRecallDict[indexModel] = 100*class_correct[i] / class_total[i]

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, "./models/model{}.ckpt".format(modelName))

print()
for i in range(1, 26):
    print("---------MODEL{}---------".format(i))
    print("Accuracy \t=> {}".format(accuracyDict[i]))
    print("Crack Recall \t=> {}".format(crackRecallDict[i]))
    print("Normal Recall \t=> {}".format(normalRecallDict[i]))
    print()
