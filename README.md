对resnet110进行l1-norm剪枝，数据集cifar10，微调时引入kl散度作为知识蒸馏损失，仅在模型输出的logits上做知识蒸馏

teacher model: resnet110

student model: pruned resnet110

$\text{total loss} = \alpha \cdot \text{KL loss} + (1 - \alpha) \cdot \text{cross entropy loss}$

训练resnet110：
```
python train_resnet.py --save_dir logs
```

剪枝resnet110：
```
python prun_resnet.py --pretrain_path logs --save_dir logs
```

微调resnet110：
```
python kd_resnet.py --pretrain_path logs --save_dir logs
```

实验结果：
|模型|剪枝比例|剪枝前acc|剪枝后acc（无知识蒸馏）|剪枝后acc（有知识蒸馏）|
|-|-|-|-|-|
|resnet110|0.5|0.934|0.927|0.929|

