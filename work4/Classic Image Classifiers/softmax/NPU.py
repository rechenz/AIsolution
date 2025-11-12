import numpy as np
import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import TensorDataset, DataLoader
import time
import psutil
import threading
import queue


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def fetch_data():
    train_data = []
    train_label = []
    test_data = []
    testlabel = []
    for i in range(5):
        data = unpickle(
            f'E:/projects/datasets/cifar-10-batches-py/data_batch_{i+1}')
        train_data.append(data[b'data'])
        train_label.append(data[b'labels'])
    test = unpickle('E:/projects/datasets/cifar-10-batches-py/test_batch')
    test_data = test[b'data']
    testlabel = test[b'labels']
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    train_data = train_data.reshape(
        (50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype("float32")
    train_data = train_data.reshape(-1, 32*32*3)
    test_data = test_data.reshape((10000, 3, 32, 32)).transpose(
        0, 2, 3, 1).astype("float32")
    test_data = test_data.reshape(-1, 32*32*3)
    return train_data, train_label, test_data, testlabel


def preprocess_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    train_data -= mean
    test_data -= mean
    return train_data, test_data


class NPUMonitor:
    def __init__(self, device):
        self.device = device
        self.monitoring = False
        self.stats_queue = queue.Queue()
        self.monitor_thread = None

    def start_monitoring(self):
        self.monitoring = True
        self.stats_queue = queue.Queue()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """持续监控NPU使用情况"""
        while self.monitoring:
            try:
                # 创建测试张量并执行计算
                test_tensor = torch.randn(1000, 1000).to(self.device)

                # 测量计算时间
                start_time = time.time()
                result = torch.matmul(test_tensor, test_tensor)

                # 确保计算完成
                if str(self.device).startswith('privateuseone'):
                    # 对于DirectML设备，使用同步操作
                    dummy = torch.tensor([1.0]).to(self.device)
                    _ = dummy + 1

                end_time = time.time()

                # 记录统计信息
                stats = {
                    'timestamp': time.time(),
                    'compute_time': end_time - start_time,
                    'memory_allocated': torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0,
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent
                }

                self.stats_queue.put(stats)
                time.sleep(0.1)  # 每100ms采样一次

            except Exception as e:
                print(f"监控错误: {str(e)}")

    def get_current_stats(self):
        """获取当前统计信息"""
        stats = []
        try:
            while True:
                stats.append(self.stats_queue.get_nowait())
        except queue.Empty:
            pass
        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_current_stats()
        if not stats:
            return

        print("\n=== NPU使用统计 ===")
        avg_compute_time = np.mean([s['compute_time'] for s in stats])
        max_compute_time = np.max([s['compute_time'] for s in stats])
        min_compute_time = np.min([s['compute_time'] for s in stats])

        print(f"平均计算时间: {avg_compute_time:.4f}秒")
        print(f"最大计算时间: {max_compute_time:.4f}秒")
        print(f"最小计算时间: {min_compute_time:.4f}秒")
        print(f"采样点数: {len(stats)}")

        if torch.cuda.is_available():
            print(f"GPU内存使用: {stats[-1]['memory_allocated']/1024**2:.1f}MB")

        print(f"CPU使用率: {stats[-1]['cpu_percent']:.1f}%")
        print(f"系统内存使用率: {stats[-1]['memory_percent']:.1f}%")


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 初始化权重
        nn.init.normal_(self.fc1.weight, std=std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=std)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epoch, monitor):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            # 每100个batch打印一次NPU统计信息
            monitor.print_stats()


def main():
    # 设置设备
    try:
        device = torch_directml.device()
        print(f"Using NPU device: {device}")
    except:
        device = torch.device('cpu')
        print("NPU not available, using CPU")

    # 创建NPU监视器
    monitor = NPUMonitor(device)

    # 获取并预处理数据
    train_data, train_label, test_data, testlabel = fetch_data()
    train_data, test_data = preprocess_data(train_data, test_data)

    # 转换为PyTorch张量
    train_data = torch.FloatTensor(train_data).to(device)
    train_label = torch.LongTensor(train_label).to(device)
    test_data = torch.FloatTensor(test_data).to(device)
    testlabel = torch.LongTensor(testlabel).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

    # 初始化模型和优化器
    model = NeuralNetwork(32*32*3, 100, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-6)

    # 开始监视NPU
    monitor.start_monitoring()

    try:
        # 训练模型
        for epoch in range(100):
            train(model, device, train_loader, optimizer, epoch, monitor)

            # 每10个epoch降低学习率
            if epoch % 10 == 0 and epoch != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

        # 评估模型
        model.eval()
        with torch.no_grad():
            output = model(train_data)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(train_label.view_as(pred)).float().mean()
            print(f'\nTrain set: Accuracy: {accuracy.item():.4f}\n')

    finally:
        # 停止监视并打印最终统计
        monitor.stop_monitoring()
        monitor.print_stats()


if __name__ == '__main__':
    main()
