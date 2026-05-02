import streamlit as st
from transformers import pipeline
from PIL import Image

# 缓存模型加载，避免每次交互都重新加载
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

def main():
    st.set_page_config(page_title="年龄分类器", page_icon="👴")
    st.title("👶 年龄分类：基于 ViT 模型")
    st.markdown("上传一张人脸照片，模型会预测年龄范围（如：`young`, `middle aged`, `senior`）")

    # 加载模型（仅第一次运行加载）
    with st.spinner("正在加载年龄分类模型，请稍候..."):
        age_classifier = load_model()

    # 文件上传组件
    uploaded_file = st.file_uploader(
        "选择一张图片（支持 jpg/png）",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # 读取并显示图片
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的图片", use_container_width=True)

        # 进行分类
        with st.spinner("正在分析年龄..."):
            predictions = age_classifier(image)

        # 按置信度降序排序
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

        # 展示最高置信度的结果
        top_pred = predictions[0]
        st.success(f"**预测年龄范围：{top_pred['label']}**")
        st.write(f"置信度：{top_pred['score']:.2%}")

        # 可选：展示所有可能类别
        with st.expander("查看所有分类结果"):
            for pred in predictions:
                st.write(f"{pred['label']}: {pred['score']:.2%}")
    else:
        st.info("👈 请先上传一张图片")

if __name__ == "__main__":
    main()
