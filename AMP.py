import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter # <--- 引入心电图机

# === A. 初始化可视化工具 ===
# 运行后会生成一个 runs/amp_experiment_1 文件夹
writer = SummaryWriter('runs/amp_experiment_1')

# === B. 数据源 (数学定义) ===
def get_good_motion(batch_size):
    """ 真实数据 (TOWR): 圆周运动 """
    # torch.rand -> 均匀分布 U[0, 1]
    t = torch.rand(batch_size, 1) * 2 * np.pi 
    # torch.cat -> 把 sin 和 cos 拼成 [batch, 2] 的矩阵
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)

def get_bad_motion(batch_size):
    """ 假数据 (RL初期): 高斯噪声 """
    # torch.randn -> 标准正态分布 N(0, 1)
    return torch.randn(batch_size, 2)

# === C. 判别器 (函数拟合器) ===
# nn.Module: 告诉 PyTorch "我是一个可以被优化的神经网络"
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential: 复合函数 f(x) = Sigmoid(Linear(ReLU(Linear(x))))
        self.net = nn.Sequential(
            nn.Linear(2, 32), # 线性变换: y = xA^T + b
            nn.ReLU(),        # 激活函数: max(0, x)
            nn.Linear(32, 1), # 再次线性变换，输出一个标量
            nn.Sigmoid()      # 压缩到 (0, 1) 区间，代表概率
        )

    def forward(self, x):
        return self.net(x)

# === D. 训练循环 ===
def run_experiment():
    D = Discriminator()
    optimizer = optim.Adam(D.parameters(), lr=0.01)
    
    # BCELoss: 二分类交叉熵
    # Loss = - [Target * log(Pred) + (1-Target) * log(1-Pred)]
    loss_fn = nn.BCELoss()

    print("开始训练... 请稍后在 TensorBoard 中查看曲线")

    for step in range(5000): # 训练 500 步
        # 1. 准备数据
        real_data = get_good_motion(64)
        fake_data = get_bad_motion(64)

        # 2. 前向传播 (判别器打分)
        pred_real = D(real_data) # 判别器看真数据 -> 应该输出 1
        pred_fake = D(fake_data) # 判别器看假数据 -> 应该输出 0

        # 3. 计算 Loss
        loss_real = loss_fn(pred_real, torch.ones_like(pred_real))   # 希望 pred_real 接近 1
        loss_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))  # 希望 pred_fake 接近 0
        total_loss = loss_real + loss_fake

        # 4. 反向传播 (梯度下降)
        optimizer.zero_grad()    # 清空之前的梯度
        total_loss.backward()    # 计算新的梯度
        optimizer.step()         # 更新参数

        # === E. 埋点记录 (关键步骤) ===
        # 把数据写入 TensorBoard
        writer.add_scalar('Loss/Total', total_loss.item(), step)
        writer.add_scalar('Score/Real_Prob', pred_real.mean().item(), step)
        writer.add_scalar('Score/Fake_Prob', pred_fake.mean().item(), step)

        if step % 50 == 0:
            print(f"Step {step}: Loss = {total_loss.item():.4f}")

    # 训练结束，关闭记录器
    writer.close()
    print("\n✅ 训练完成！")
    print("请在终端运行以下命令查看图表：")
    print("tensorboard --logdir=runs")

if __name__ == "__main__":
    run_experiment()