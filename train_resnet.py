import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from model.resnet import *
import torch.optim as optim

# 参数配置
parser = argparse.ArgumentParser(description='PyTorch Cifar10 ResNet56 Training Without Pruning')
parser.add_argument('--save_dir', help='Folder to save checkpoints and log.')
BATCH_SIZE=256
LR=0.1
EPOCHS=200

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using gpu")
else:
    print("warning: using cpu")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪32x32，四周填充4像素
    transforms.RandomHorizontalFlip(),     # 50%概率水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10官方均值和方差
])

# 测试集：仅做归一化（保持原始数据分布）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
criterion = nn.CrossEntropyLoss()
model = ResNet56().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)],  # 50%和75%处衰减
    gamma=0.1  # 衰减系数
)

def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, f'model_best.pth')
    init_path=os.path.join(args.save_dir,f'init.pth')
    torch.save(model.state_dict(), init_path)
    best_accuracy = 0  # 2 初始化best test accuracy
    print("start traing resnet56 on cifar10, total epochs=",EPOCHS)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # 每个epoch结束后测试
        train_acc = 100 * correct / total
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        print(f'Epoch {epoch + 1} - train accuracy: {train_acc:.2f}%  test accuracy: {test_acc:.2f}%')

        # 仅在当前测试准确率高于最佳准确率时保存模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), save_path)

        scheduler.step()
        model.train()
    print("finish training, best accuracy=%.3f\n"%best_accuracy)
if __name__ == "__main__":
    main()


