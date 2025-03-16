import torch
import os
from model_loader import DualEyeNet
import sys

class ONNXCompatibleModel(torch.nn.Module):
    """包装模型使其与ONNX兼容"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, left_eye, right_eye):
        # 创建模型期望的字典格式
        batch = {'left_eye': left_eye, 'right_eye': right_eye}
        return self.model(batch)

def convert_pytorch_to_onnx():
    # 强制使用CPU进行ONNX导出，避免设备不匹配问题
    device = torch.device('cpu')
    
    # 设置模型路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # 确保models目录存在
    os.makedirs(models_dir, exist_ok=True)
    
    pytorch_model_path = os.path.join(models_dir, "best_model.pth")
    onnx_model_path = os.path.join(models_dir, "best_model.onnx")
    
    print(f"PyTorch模型路径: {pytorch_model_path}")
    print(f"ONNX模型将保存到: {onnx_model_path}")
    
    # 加载模型
    model = DualEyeNet(num_classes=8)
    
    try:
        # 加载模型权重
        checkpoint = torch.load(pytorch_model_path, map_location=device)
        
        # 检查是否是完整的检查点或只是模型状态字典
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"模型成功加载自: {pytorch_model_path}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        sys.exit(1)
    
    # 确保模型在CPU上
    model = model.to(device)
    model.eval()
    
    # 创建ONNX兼容的包装模型
    onnx_model = ONNXCompatibleModel(model)
    
    # 创建示例输入 - 确保在CPU上
    batch_size = 1
    channels = 3
    height = 380
    width = 380
    
    dummy_left_eye = torch.randn(batch_size, channels, height, width, device=device)
    dummy_right_eye = torch.randn(batch_size, channels, height, width, device=device)
    
    # 导出ONNX模型
    try:
        torch.onnx.export(
            onnx_model,                           # 包装后的模型
            (dummy_left_eye, dummy_right_eye),    # 模型输入（作为元组）
            onnx_model_path,                      # 输出路径
            export_params=True,                   # 存储训练好的参数权重
            opset_version=12,                     # ONNX版本
            do_constant_folding=True,             # 是否执行常量折叠优化
            input_names=['left_eye', 'right_eye'],# 输入名称
            output_names=['output'],              # 输出名称
            dynamic_axes={                        # 动态尺寸
                'left_eye': {0: 'batch_size'},
                'right_eye': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX模型已成功保存到: {onnx_model_path}")
        
        # 验证ONNX模型
        import onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型检查通过")
        
        # 打印模型大小
        pytorch_size = os.path.getsize(pytorch_model_path) / (1024 * 1024)
        onnx_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        print(f"PyTorch模型大小: {pytorch_size:.2f} MB")
        print(f"ONNX模型大小: {onnx_size:.2f} MB")
        if onnx_size > 0:
            print(f"压缩比例: {pytorch_size/onnx_size:.2f}x")
        
    except Exception as e:
        print(f"导出ONNX模型时出错: {e}")
        print(f"错误详情: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    convert_pytorch_to_onnx() 