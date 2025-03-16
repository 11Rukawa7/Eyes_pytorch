import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import sys
import os
import onnxruntime as ort  # 确保导入onnxruntime
# 设置默认编码为UTF-8
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='UTF-8')

# 疾病名称映射
DISEASE_NAMES = {
    0: '正常',
    1: '糖尿病',
    2: '青光眼',
    3: '白内障',
    4: 'AMD',
    5: '高血压',
    6: '近视',
    7: '其他'
}

# 融合模块
class SEFusion(nn.Module):
    """使用SE注意力机制的特征融合"""
    
    def __init__(self, in_channels: int = 1792):  # EfficientNet-B4的输出通道数
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([left_feat, right_feat], dim=1)
        weights = self.se(combined)
        w1, w2 = torch.chunk(weights, 2, dim=1)
        fused = w1 * left_feat + w2 * right_feat
        return fused

class CrossAttention(nn.Module):
    """交叉空间注意力"""
    
    def __init__(self, in_channels: int = 1792):  # EfficientNet-B4的输出通道数
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.size()
        
        # 生成查询、键和值
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

# 主模型
class DualEyeNet(nn.Module):
    """双眼底图像多标签分类模型"""
    
    def __init__(
        self,
        num_classes: int = 8,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # 使用预训练的EfficientNet作为backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # 冻结backbone的前15层参数
        if freeze_backbone:
            for param in list(self.backbone.parameters())[:15]:
                param.requires_grad = False
        
        # 特征通道数
        self._feature_channels = 1792  # EfficientNet-B4的输出通道数
        
        # SE融合模块
        self.se_fusion = SEFusion(in_channels=self._feature_channels)
        # 跨模态注意力模块
        self.cross_attention = CrossAttention(in_channels=self._feature_channels)
        
        # 分类器
        self.classifier = nn.Sequential(
            # 将特征图尺寸调整为1x1
            nn.AdaptiveAvgPool2d(1),
            # 展平特征图
            nn.Flatten(),
            # 全连接层，输入通道数为特征通道数，输出通道数为512
            nn.Linear(self._feature_channels, 512),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # Dropout层，丢弃率为0.3
            nn.Dropout(0.3),
            # 全连接层，输入通道数为512，输出通道数为num_classes
            nn.Linear(512, num_classes)
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 提取左眼特征
        left_features = self.backbone.extract_features(batch['left_eye'])
        # 提取右眼特征
        right_features = self.backbone.extract_features(batch['right_eye'])
        
        # SE融合
        fused_features = self.se_fusion(left_features, right_features)
        # 跨模态注意力
        enhanced_features = self.cross_attention(fused_features)
        
        # 分类
        logits = self.classifier(enhanced_features)
        
        return logits

def get_transforms():
    """获取图像预处理变换"""
    return A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

class EyeDiagnosisModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_onnx = model_path.endswith('.onnx')
        self.model = self._load_model(model_path)
        self.transforms = get_transforms()
        
    def _load_model(self, model_path):
        """加载模型，支持PyTorch和ONNX格式"""
        # 确保路径处理正确
        if not os.path.exists(model_path):
            # 尝试相对路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        if self.is_onnx:
            try:
                # 为ONNX创建推理会话
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                session = ort.InferenceSession(model_path, providers=providers)
                print(f"ONNX模型成功加载自: {model_path}")
                return session
            except Exception as e:
                print(f"加载ONNX模型时出错: {e}")
                raise
        else:
            # 加载PyTorch模型
            model = DualEyeNet(num_classes=8)
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 检查是否是完整的检查点或只是模型状态字典
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                print(f"PyTorch模型成功加载自: {model_path}")
            except Exception as e:
                print(f"加载PyTorch模型时出错: {e}")
                raise
                
            model = model.to(self.device)
            model.eval()
            return model
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 确保图像是RGB格式
        if len(image.shape) == 2:  # 灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA图像
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # 应用变换
        transformed = self.transforms(image=image)
        return transformed['image']
    
    def predict(self, left_eye_img, right_eye_img):
        """预测眼底图像的疾病"""
        # 预处理图像
        left_tensor = self.preprocess_image(left_eye_img)
        right_tensor = self.preprocess_image(right_eye_img)
        
        if self.is_onnx:
            # ONNX模型推理
            # 转换为numpy数组并确保正确的形状
            left_np = left_tensor.numpy()
            right_np = right_tensor.numpy()
            
            # 添加批次维度 - 修复维度错误
            if len(left_np.shape) == 3:
                left_np = np.expand_dims(left_np, axis=0)
            if len(right_np.shape) == 3:
                right_np = np.expand_dims(right_np, axis=0)
            
            # 创建输入字典
            ort_inputs = {
                'left_eye': left_np,
                'right_eye': right_np
            }
            
            # 运行推理
            ort_outputs = self.model.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            # 应用sigmoid获取概率
            probs = 1 / (1 + np.exp(-logits))[0]
        else:
            # PyTorch模型推理
            # 添加批次维度
            left_tensor = left_tensor.unsqueeze(0).to(self.device)
            right_tensor = right_tensor.unsqueeze(0).to(self.device)
            
            # 构建输入字典
            inputs = {
                'left_eye': left_tensor,
                'right_eye': right_tensor
            }
            
            # 进行预测
            with torch.no_grad():
                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
            
            # 将预测结果转换为CPU并转为NumPy数组
            probs = probs.cpu().numpy()[0]
            
        # 返回每个类别的预测概率
        results = []
        for i, prob in enumerate(probs):
            results.append({
                'disease': DISEASE_NAMES[i],
                'probability': float(prob),
                'predicted': prob > 0.5
            })
            
        return results