import torch
import torch.nn.functional as F
import torch.optim as optim
from model.resnet110 import *
import argparse
import os
import torchvision.transforms as transforms
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using gpu")
else:
    print("warning: using cpu")

parser = argparse.ArgumentParser(description='l1-norm-pruning resnet56 on cifar10')
parser.add_argument('--pretrain_path', help='教师模型权重所在的文件夹')
parser.add_argument('--save_dir', help='微调后的模型权重存储的文件路径')
args = parser.parse_args()

BATCH_SIZE = 256
EPOCHS = 200
teacher_pth = os.path.join(args.pretrain_path, f'model_best.pth')
student_pth = os.path.join(args.pretrain_path, f'pruned.pth')
retrained_pth = os.path.join(args.save_dir, f'retrained.pth')

# 教师模型加载预训练的权重
teacher_model = ResNet110().to(device)
teacher_model.load_state_dict(torch.load(teacher_pth, map_location=device, weights_only=False))
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 学生模型直接加载结构+权重
student_model = torch.load(student_pth, map_location=device, weights_only=False)
student_model = student_model.to(device)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

print("teacher model accuracy: %.3f"%evaluate_accuracy(teacher_model,testloader,device))
print("before training student model accuracy: %.3f"%evaluate_accuracy(student_model,testloader,device))

# -------------------------- 知识蒸馏训练 --------------------------
T = 6  # 温度参数
alpha = 0.95  # KL散度权重
optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)],  # 50%和75%处衰减
    gamma=0.1  # 衰减系数
)

print("start training with knowledge distillation")
best_test_acc = 0.0

for epoch in range(EPOCHS):
    student_model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for data in trainloader:
        images, labels = data[0].to(device), data[1].to(device)
        # 教师模型不更新参数
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        # 学生模型更新参数
        student_logits = student_model(images)
        # KL散度损失
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        student_soft = F.log_softmax(student_logits / T, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        # 交叉熵损失
        hard_loss = F.cross_entropy(student_logits, labels)
        # 总损失=学生模型和教师模型的kl散度+学生模型和真实标签的交叉熵损失
        total_loss = alpha * kl_loss + (1 - alpha) * hard_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        _, predicted = torch.max(student_logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        running_loss += total_loss.item()
    train_acc = 100 * train_correct / train_total
    avg_loss = running_loss / len(trainloader)
    student_model.eval()
    test_acc = evaluate_accuracy(student_model, testloader, device)
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] - 平均损失：{avg_loss:.4f} | 训练集准确率：{train_acc:.2f}% | 测试集准确率：{test_acc:.2f}%")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        #同时保存模型参数和结构
        torch.save(student_model, retrained_pth)
    scheduler.step()

# 训练结束后打印总结
print("finish training with knowledge distillation")
print(f"最佳测试集准确率：{best_test_acc:.2f}%")
