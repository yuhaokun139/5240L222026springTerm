pip install transformers torch torchaudio pillow
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_gender_pipeline():
    """加载年龄与性别预测模型并缓存"""
    return pipeline("image-classification", model="abhilash88/age-gender-prediction")

def identify_gender_from_image(image_path):
    """从图像中识别性别，返回性别标签"""
    try:
        # 确保图像是 RGB 格式
        image = Image.open(image_path).convert("RGB")
        # 加载 Pipeline
        gender_pipe = load_gender_pipeline()
        # 生成预测
        predictions = gender_pipe(image)
        # 提取性别信息
        # 预测结果通常是一个列表，包含 'label' 和 'score'
        # 例如：[{'label': 'Female', 'score': 0.95}, ...]
        if predictions:
            # 获取置信度最高的结果
            top_prediction = predictions[0]
            gender = top_prediction['label']
            confidence = top_prediction['score']
            return gender, confidence
        else:
            return "Unknown", 0.0
    except Exception as e:
        st.error(f"识别性别时出错: {e}")
        return "Error", 0.0

def generate_audio_story(image_path, gender, confidence):
    """
    根据图像生成个性化的音频故事
    :param image_path: 图像文件路径
    :param gender: 识别出的性别
    :param confidence: 识别置信度
    """
    # 1. 生成图像描述 (基于图像内容的通用描述)
    # ... (你原有的图像描述代码，例如使用 BLIP 模型)
    description = "我看到一张照片，里面是..." # 这是你之前的通用描述

    # 2. 将性别信息融入故事文本
    if confidence > 0.7:
        story = f"这是一位{gender}，{description}。"
    else:
        # 如果置信度不高，可以保持中性
        story = f"照片中的人，{description}。"

    # 3. 文本转音频 (你之前定义的 text2audio 函数)
    # audio_data = text2audio(story)
    # return audio_data
    # ... (你的后续 TTS 逻辑)
