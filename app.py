import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from model_loader import EyeDiagnosisModel
from utils import load_image_from_bytes, resize_image_for_display

# 设置默认编码为UTF-8
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='UTF-8')
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="眼底图像疾病诊断系统",
    page_icon="👁️",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    """加载模型，优先使用ONNX模型"""
    model_path = "models/best_model.onnx"
    
    # 检查ONNX模型是否存在
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, model_path)
    
    if os.path.exists(full_path):
        st.info("正在加载ONNX模型...")
        return EyeDiagnosisModel(model_path)
    else:
        st.warning("ONNX模型不存在，尝试加载PyTorch模型...")
        return EyeDiagnosisModel("models/best_model.onnx")

# 创建应用标题
st.title("👁️ 眼底图像疾病诊断系统")
st.markdown("上传左右眼底图像，系统将自动分析可能的眼部疾病。")

# 创建两列布局
col1, col2 = st.columns(2)

# 左眼图像上传
with col1:
    st.header("左眼图像")
    left_eye_file = st.file_uploader("上传左眼图像", type=["jpg", "jpeg", "png"])
    
    if left_eye_file is not None:
        left_eye_bytes = left_eye_file.getvalue()
        left_eye_img = load_image_from_bytes(left_eye_bytes)
        st.image(left_eye_img, caption="左眼图像", use_container_width=True)
    else:
        st.info("请上传左眼图像")
        left_eye_img = None

# 右眼图像上传
with col2:
    st.header("右眼图像")
    right_eye_file = st.file_uploader("上传右眼图像", type=["jpg", "jpeg", "png"])
    
    if right_eye_file is not None:
        right_eye_bytes = right_eye_file.getvalue()
        right_eye_img = load_image_from_bytes(right_eye_bytes)
        st.image(right_eye_img, caption="右眼图像", use_container_width=True)
    else:
        st.info("请上传右眼图像")
        right_eye_img = None

# 诊断按钮
if st.button("开始诊断", disabled=(left_eye_img is None or right_eye_img is None)):
    with st.spinner("正在分析图像..."):
        try:
            # 加载模型
            model = load_model()
            
            # 进行预测
            results = model.predict(left_eye_img, right_eye_img)
            
            # 显示结果
            st.header("诊断结果")
            
            # 创建两列：一列显示表格，一列显示图表
            res_col1, res_col2 = st.columns([3, 2])
            
            with res_col1:
                # 创建结果表格
                result_data = {
                    "疾病": [],
                    "概率": [],
                    "诊断结果": []
                }
                
                for result in results:
                    result_data["疾病"].append(result["disease"])
                    result_data["概率"].append(f"{result['probability']:.2%}")
                    result_data["诊断结果"].append("✅ 检测到" if result["predicted"] else "❌ 未检测到")
                
                st.table(result_data)
            
            with res_col2:
                # 创建条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                
                diseases = [r["disease"] for r in results]
                probs = [r["probability"] for r in results]
                colors = ['green' if p > 0.5 else 'gray' for p in probs]
                
                ax.barh(diseases, probs, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('概率')
                ax.set_title('疾病检测概率')
                
                # 添加概率值标签
                for i, v in enumerate(probs):
                    ax.text(v + 0.01, i, f"{v:.2f}", va='center')
                
                st.pyplot(fig)
            
            # 显示诊断总结
            st.subheader("诊断总结")
            
            detected_diseases = [r["disease"] for r in results if r["predicted"]]
            
            if "Normal" in detected_diseases and len(detected_diseases) > 1:
                detected_diseases.remove("Normal")
                
            if len(detected_diseases) == 0:
                st.success("👍 未检测到任何眼部疾病。")
            elif "Normal" in detected_diseases and len(detected_diseases) == 1:
                st.success("👍 眼部状况正常，未检测到任何疾病。")
            else:
                st.warning(f"⚠️ 检测到可能的眼部疾病: {', '.join(detected_diseases)}")
                st.info("请注意：此结果仅供参考，建议咨询专业眼科医生进行进一步诊断。")
                
        except Exception as e:
            st.error(f"诊断过程中出错: {str(e)}")

# 添加页脚
st.markdown("---")
st.markdown("👁️ **眼底图像疾病诊断系统** | 基于深度学习的眼底图像分析")
st.markdown("⚠️ 免责声明：本系统仅供研究和参考，不应替代专业医疗诊断。")
