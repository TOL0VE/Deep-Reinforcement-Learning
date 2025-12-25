import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

# === 0. 自动检测 GPU 设备 ===
# 如果有 NVIDIA 显卡且安装了 CUDA，就使用 'cuda'，否则使用 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 当前使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️ 未检测到 GPU，正在使用 CPU 慢速运行...")

# === A. 初始化可视化工具 ===
writer = SummaryWriter('runs/amp_gpu_experiment')

# === B. 数据源 (需要搬运到 device) ===
def get_good_motion(batch_size):
    """ 真实数据 (TOWR): 圆周运动 """
    t = torch.rand(batch_size, 1) * 2 * np.pi #batch_size x 1
    data = torch.cat([torch.sin(t), torch.cos(t)], dim=1)# batch_size x 2
    # 【关键】把数据搬到 GPU
    return data.to(device)

def get_bad_motion(batch_size):
    """ 假数据 (RL初期): 高斯噪声 """
    data = torch.randn(batch_size, 2)# batch_size x 2
    # 【关键】把数据搬到 GPU
    return data.to(device)

# === C. 判别器 (需要搬运到 device) ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), # 加大一点网络看看 GPU 威力
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === D. 训练循环 ===
def run_experiment():
    # 1. 实例化网络
    D = Discriminator()
    
    # 【关键】把整个神经网络模型搬到 GPU
    D = D.to(device)
    
    optimizer = optim.Adam(D.parameters(), lr=0.001)
    '''D.parameters(): 这就是公式里的 $\theta$（我们需要优化的变量：权重 $W$ 和偏置 $b$）。
    lr=0.001: 这就是基础步长 $\alpha$。
    比刚开始学的SGD优化器更高级，但本质是一样的。
    '''

    loss_fn = nn.BCELoss()

    print("开始极速训练...")
    start_time = time.time()

    # 增加训练步数，体现 GPU 优势
    total_steps = 100000
    
    for step in range(total_steps):
        # 1. 准备数据 (已经在 get_motion 函数里 to(device) 了)
        real_data = get_good_motion(1024) # 加大 Batch Size 榨干显卡
        fake_data = get_bad_motion(1024)

        # 2. 前向传播
        pred_real = D(real_data)
        pred_fake = D(fake_data)

        # 3. 计算 Loss
        loss_real = loss_fn(pred_real, torch.ones_like(pred_real))
        loss_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))
        total_loss = loss_real + loss_fake

        # 4. 反向传播
        optimizer.zero_grad()
        '''字面意思：把梯度的积累清零。
        为什么要这样做？
        在 PyTorch 的设计中，.grad 是累加的（Accumulated）。
        如果你不加这行，第 1 步算出的梯度是 $g_1$。
        第 2 步算出的梯度是 $g_2$，PyTorch 会把它加到原来的上面，变成 $g_1 + g_2$。
        第 3 步变成 $g_1 + g_2 + g_3$。
        这在 RNN 这种特殊网络里有用，但在我们这里是大忌！我们希望每一步的梯度只代表当下的方向。
        
        类比：这就好比你要称重。每次称重前，都要把秤归零，否则下一次称的就是两个物体的总重了。'''
        total_loss.backward()# 计算新的梯度
        optimizer.step()# theta t+1 = ********

        # === E. 埋点记录 ===
        # 注意：写入 TensorBoard 时，通常需要把数据从 GPU 拉回 CPU (.item() 会自动处理，但如果是 tensor 就要 .cpu())
        if step % 100 == 0:
            # 记录到 TensorBoard
            writer.add_scalar('Loss/Total', total_loss.item(), step)
            writer.add_scalar('Score/Real_Prob', pred_real.mean().item(), step)
            writer.add_scalar('Score/Fake_Prob', pred_fake.mean().item(), step)
            
            print(f"Step {step}: Loss = {total_loss.item():.4f}")

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 50)
    print(f"✅ 训练完成！")
    print(f"📊 总耗时: {duration:.2f} 秒")
    print(f"🚀 平均速度: {total_steps / duration:.1f} steps/sec")
    print("-" * 50)
    
    writer.close()

if __name__ == "__main__":
    run_experiment()