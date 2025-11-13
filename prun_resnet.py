import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.resnet import *
from copy import deepcopy
import torch.optim as optim

parser = argparse.ArgumentParser(description='l1-norm-pruning resnet56 on cifar10')
parser.add_argument('--pretrain_path', help='需要剪枝的模型权重所在的文件夹')
parser.add_argument('--save_dir', help='剪枝后的模型权重存储的文件路径')
args = parser.parse_args()

def count_parameters(model):
    """计算模型总参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using gpu")
else:
    print("warning: using cpu")

BATCH_SIZE=256
PRUN_RATIO=0.2
pretrain_pth=os.path.join(args.pretrain_path, f'model_best.pth')
init_pth=os.path.join(args.pretrain_path,f'init.pth')
retrained_pth=os.path.join(args.save_dir,f'retrained.pth')
original_model = ResNet56().to(device)
original_model.load_state_dict(torch.load(pretrain_pth))
pruned_model = ResNet56().to(device)
pruned_model.load_state_dict(torch.load(init_pth))

# 计算剪枝前参数数量
before_params = count_parameters(original_model)
print(f"剪枝前参数数量: {before_params:,}")

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

#查看预训练模型的准确率（剪枝前）
original_model.eval()
total=0
correct=0
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = original_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("before pruning, test accuracy is %.3f"% (100 * correct / total))

#l1-norm剪枝，结构化剪枝
in_channel=3
masks=[]
ratio=PRUN_RATIO

layer_id=0
masks=[]
for m in original_model.modules():
    if isinstance(m,nn.Conv2d) and m.weight.data.shape[2]==3:
        if layer_id % 2 == 0:
            # 偶数层：保留全部通道，生成 0~out_channel-1 的索引mask
            out_channel=m.weight.data.shape[0]
            keep_indices = torch.arange(out_channel, device=device)  # 格式：[0,1,2,...,out_channel-1]
            masks.append(keep_indices)
            layer_id += 1
            continue
        #通用剪枝逻辑，剪输出通道,只对残差块的第一个卷积层进行剪枝
        out_channel=m.weight.data.shape[0]
        prun_num=int(out_channel*ratio)
        filter_l1_norm = m.weight.data.norm(p=1, dim=(1, 2, 3))
        prune_filter_indices = torch.argsort(filter_l1_norm)[:prun_num]
        keep_mask = torch.ones(out_channel, dtype=torch.bool, device=device)
        keep_mask[prune_filter_indices] = False
        keep_indices = torch.where(keep_mask)[0]
        masks.append(keep_indices)
        layer_id+=1

layer_id=0
in_mask=torch.tensor([0, 1, 2], device=device)
out_mask=masks[layer_id]
for m in pruned_model.modules():
    if isinstance(m,nn.Conv2d) and m.weight.data.shape[2]==3:
        out_mask = masks[layer_id]
        keep_out_channel = len(out_mask)
        keep_in_channel = len(in_mask)
        m.weight.data = m.weight.data[out_mask].clone()
        if m.bias is not None:
            m.bias.data = m.bias.data[out_mask].clone()
        #剪输入通道
        m.weight.data = m.weight.data[:, in_mask].clone()
        m.in_channels = keep_in_channel
        m.out_channels = keep_out_channel
        in_mask=out_mask
        layer_id+=1
    elif isinstance(m, nn.BatchNorm2d):
        #BN层的输入mask就是上一个卷积层的输出mask
        bn_keep_indices = in_mask
        m.weight.data = m.weight.data[bn_keep_indices].clone()
        m.bias.data = m.bias.data[bn_keep_indices].clone()
        m.running_mean = m.running_mean[bn_keep_indices].clone()
        m.running_var = m.running_var[bn_keep_indices].clone()
        m.num_features = len(bn_keep_indices)


pruned_model.eval()
total=0
correct=0
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = pruned_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("after pruning, test accuracy is %.3f"% (100 * correct / total))
after_params = count_parameters(pruned_model)
print(f"剪枝后参数数量: {after_params:,}")
#torch.save(pruned_model.state_dict(), pruned_pth)

print("finetuning after pruning")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pruned_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
EPOCHS = 200
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)],  # 50%和75%处衰减
    gamma=0.1  # 衰减系数
)

# 训练参数
best_accuracy = 0.0  # 初始化最佳准确率
pruned_model.train()  # 切换到训练模式
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = pruned_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # 每个epoch结束后测试
    train_acc = 100 * correct / total
    pruned_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = pruned_model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total
    print(f'Epoch {epoch + 1} - train accuracy: {train_acc:.2f}%  test accuracy: {test_acc:.2f}%')

    # 仅在当前测试准确率高于最佳准确率时保存模型
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(pruned_model.state_dict(), retrained_pth)

    scheduler.step()
    pruned_model.train()

print(f"after finetuning, best accuracy: {best_accuracy:.2f}%")























