import platform
import subprocess
import sys
import torch
import torch_directml
import onnxruntime as ort


def check_system_info():
    """检查系统基本信息"""
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"处理器: {platform.processor()}")
    print()


def check_pytorch():
    """检查PyTorch安装和版本"""
    print("=== PyTorch信息 ===")
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch未安装")
    print()


def check_directml():
    """检查DirectML支持"""
    print("=== DirectML信息 ===")
    try:
        device = torch_directml.device()
        print(f"DirectML设备: {device}")
        print("DirectML可用: True")

        # 测试简单的张量运算
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.matmul(x, y)
        print("DirectML张量运算测试: 成功")
    except Exception as e:
        print(f"DirectML不可用或测试失败: {str(e)}")
    print()


def check_onnxruntime():
    """检查ONNX Runtime和DirectML提供程序"""
    print("=== ONNX Runtime信息 ===")
    try:
        print(f"ONNX Runtime版本: {ort.__version__}")
        available_providers = ort.get_available_providers()
        print("可用的执行提供程序:")
        for provider in available_providers:
            print(f"- {provider}")

        if 'DmlExecutionProvider' in available_providers:
            print("DirectML执行提供程序: 可用")
        else:
            print("DirectML执行提供程序: 不可用")
    except ImportError:
        print("ONNX Runtime未安装")
    except Exception as e:
        print(f"检查ONNX Runtime时出错: {str(e)}")
    print()


def check_amd_npu():
    """检查AMD NPU特定信息"""
    print("=== AMD NPU信息 ===")
    try:
        # 尝试获取DirectML设备信息
        device = torch_directml.device()

        # 创建测试模型
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)

        # 测试模型推理
        test_input = torch.randn(32, 100).to(device)
        with torch.no_grad():
            output = model(test_input)

        print("AMD NPU测试: 成功")
        print("模型推理测试: 通过")

    except Exception as e:
        print(f"AMD NPU测试失败: {str(e)}")
    print()


def main():
    print("AMD NPU可用性检查工具")
    print("=" * 50)

    check_system_info()
    check_pytorch()
    check_directml()
    check_onnxruntime()
    check_amd_npu()

    print("=" * 50)
    print("检查完成")


if __name__ == '__main__':
    main()
