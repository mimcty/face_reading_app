import streamlit as st
import requests
from PIL import Image
from datetime import datetime
from io import BytesIO
import time


# ========== Utility Styling Functions ==========

def local_css(file_name):
    """Load local CSS file and apply custom styles"""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def set_background():
    """Set the high-tech background with subtle Chinese elements"""

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap');
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
            font-family: 'JetBrains Mono', monospace;
            background: url('https://media.istockphoto.com/id/1151082661/vector/retro-sci-fi-background-futuristic-landscape-of-the-80s-digital-cyber-surface-suitable-for.jpg?s=612x612&w=0&k=20&c=4HbMZEmxF08zcS_NgSXDKBJXsWSZTAXRKuC1UNvlOQY=');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
        }

        body, h1, h2, h3, h4, h5, h6, span, .st.button {
            font-family: 'JetBrains Mono', monospace;
        }
        
        p, .st.button, label {
            font-size: 20px;
        }
        
            h1 {
        font-size: 3.5rem;
        }
        h2 {
            font-size: 2.5rem;
        }
        h3 {
            font-size: 2rem;
        }
        h4 {
            font-size: 1.5rem;
        }
        h5, h6 {
            font-size: 1.2rem;
        }

        /* Add subtle Chinese pattern overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            opacity: 0.03;
            pointer-events: none;
            z-index: -1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ========== Premium Experience Elements ==========

def show_loading_animation():
    steps = [
        "识别五官比例 / Analyzing facial structure...",
        "提取五行特征 / Mapping elemental traits...",
        "综合相学模型 / Combining TCM face reading...",
        "生成命理报告 / Generating full analysis...",
    ]
    for i, step in enumerate(steps):
        st.info(step)
        st.progress((i + 1) / len(steps))
        time.sleep(0.7)


def display_element_section(measurements):
    st.markdown("### 🪐 五行命理分析 / Elemental Analysis")
    col1, col2 = st.columns(2)
    element_data = measurements["chinese_reading"]

    with col1:
        st.markdown(f"**主导五行 / Dominant Element:** `{element_data['element']}`")
        st.markdown(f"**面型格局 / Face Shape:** `{element_data['shape']}`")

    with col2:
        st.markdown("**五行得分 / Elemental Scores:**")
        st.json(element_data["scores"])

    st.markdown("**面相特征判断依据 / Breakdown:**")
    for feature, (elem, reason) in element_data["breakdown"].items():
        st.markdown(f"- `{feature}`: {reason} (属{elem})")


def display_professional_analysis(analysis):
    section_titles = {
        "three_divisions": "三停五岳总论 / Facial Zones Overview",
        "facial_features": "五官精微分析 / Feature Analysis",
        "twelve_palaces": "十二宫详批 / Palace Insights",
        "element_analysis": "五行命理精解 / Elemental Interpretation",
        "enhancement": "改运秘法 / Enhancement Advice"
    }

    for key, title in section_titles.items():
        if key in analysis:
            with st.expander(title, expanded=True if key in ["three_divisions", "element_analysis"] else False):
                st.markdown(analysis[key])
        else:
            st.warning(f"{title} 不可用 / Not available.")


def display_life_predictions(predictions):
    st.markdown("### 🔮 生命阶段解读 / Life Stage Analysis")
    life = predictions.get("life_stages", {})
    for stage, content in life.items():
        name = {
            "early_life": "上停 / Early Life (0–28)",
            "mid_life": "中停 / Mid Life (28–50)",
            "late_life": "下停 / Later Life (50+)"
        }.get(stage, stage)
        st.subheader(name)
        st.markdown(content.get("content", "无内容"))

    if predictions.get("age_ranges"):
        st.subheader("📆 年龄段运势 / Age Predictions")
        for pred in predictions["age_ranges"]:
            st.markdown(f"- `{pred['age_range']}`: {pred['prediction']}")

    if predictions.get("key_years"):
        st.subheader("📅 特定年份预测 / Key Year Predictions")
        for pred in predictions["key_years"]:
            st.markdown(f"- `{pred['year']}`: {pred['prediction']}")


def display_partner_tab(result):
    partner_data = result.get("ideal_partner", {})
    if not partner_data:
        st.warning("尚未生成理想伴侣数据。")
        if st.button("生成理想伴侣分析"):
            with st.spinner("正在生成伴侣图像与分析..."):
                response = requests.post("http://localhost:8000/generate_partner", json=result)
                if response.status_code == 200:
                    st.success("生成完成，请重新运行查看结果。")
                    st.rerun()
                else:
                    st.error(f"生成失败: {response.text}")
        return

    st.markdown("### 👫 理想伴侣分析 / Ideal Partner Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**伴侣面相图像 / Portrait:**")
        if "image_url" in partner_data:
            try:
                response = requests.get(partner_data["image_url"])
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="📥 下载伴侣画像",
                    data=buf.getvalue(),
                    file_name="ideal_partner.png",
                    mime="image/png"
                )
            except:
                st.error("无法加载伴侣图像")
        else:
            st.warning("图像数据缺失")

    with col2:
        st.markdown("**配对解释 / Match Explanation:**")
        st.markdown(partner_data.get("analysis", "无配对分析"))

    st.markdown("### 🔗 相合分析 / Compatibility")
    for item in partner_data.get("compatibility", []):
        st.markdown(f"""
        - **方面 / Aspect:** {item['aspect']}
        - **分析 / Analysis:** {item['analysis']}
        - **建议 / Advice:** {item['advice']}
        """)


# ========== Main Application ==========

API_ENDPOINT = "http://localhost:8000/analyze"

def main():
    set_background()
    local_css("style.css")

    st.markdown(
        """
        <div style="text-align:center; 
                    padding:30px; 
                    margin:20px 0 40px 0;
                    border-radius:12px;
                    background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
                    border: 1px solid #D4AF37;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.3),
                                inset 0 0 15px rgba(212,175,55,0.2);
                    position: relative;">
            <h1 style="color:#D4AF37; 
                       margin:0; 
                       font-size:3.5rem;
                       text-shadow: 0 0 10px rgba(212,175,55,0.5);">面相匹配</h1>
            <h2 style="color:#D4AF37; 
                       margin:0; 
                       font-size:2rem;
                       letter-spacing: 2px;">Chinese Face Reading Matchmaker</h2>
            <div style="height: 2px; 
                       background: linear-gradient(90deg, transparent 0%, #D4AF37 50%, transparent 100%);
                       margin: 15px auto;
                       width: 60%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
                    border-radius: 12px;
                    padding: 25px;
                    border: 1px solid #D4AF37;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <h2 style="color:#D4AF37;">About This App</h2>
            <p style="color:#F5DEB3;">This app uses ancient Chinese face reading principles (面相学) to analyze your facial features and
            suggest your ideal compatible partner based on:</p>
            <ul style="color:#F5DEB3;">
                <li>Five Element Theory (五行)</li>
                <li>Facial Feature Analysis</li>
                <li>Personality Compatibility</li>
            </ul>
            <p style="color:#F5DEB3;">Based on your Chinese facial reading analysis, a photo of your ideal partner will be generated. Your
            partner will have features that complement yours, alongside personality and element comparison.
            This depiction is AI-generated and experimental only.</p>
        </div>
        """, unsafe_allow_html=True)
    # Main Content - Two column layout
    col1, col2 = st.columns([1, 1], gap="large")

    # Left Column - Photo Upload with modern card design
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
                    border-radius: 12px;
                    padding: 25px;
                    margin-bottom: 20px;
                    border: 1px solid #D4AF37;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <h3 style="text-align: center; 
                      color: #D4AF37;
                      margin-bottom: 20px;">上传你的照片</h3>
            <h3 style="text-align: center; 
                      color: #D4AF37;
                      margin-top: 0;">UPLOAD YOUR PHOTO</h4>
        </div>
        """, unsafe_allow_html=True)

        # Modern file uploader styling
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"],
                                         label_visibility="collapsed",
                                         key="file_uploader")

        if uploaded_file:
            # Display uploaded image with modern frame
            img = Image.open(uploaded_file)
            st.markdown("""
            <div style="border: 2px solid #D4AF37;
                        border-radius: 8px;
                        padding: 10px;
                        background: rgba(15,12,41,0.7);
                        box-shadow: 0 0 15px rgba(212,175,55,0.3);
                        margin-top: 20px;
                        display: flex;
                        justify-content: center;">
            """, unsafe_allow_html=True)

            st.image(img, caption="Your Face", width=300,
                     use_column_width=True, output_format="JPEG")

            st.markdown("</div>", unsafe_allow_html=True)

    # Right Column - Preferences with modern card design
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
                    border-radius: 12px;
                    padding: 25px;
                    margin-bottom: 20px;
                    border: 1px solid #D4AF37;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <h3 style="text-align: center; 
                      color: #D4AF37;
                      margin-bottom: 20px;">伴侣偏好</h3>
            <h4 style="text-align: center; 
                      color: #D4AF37;
                      margin-top: 0;">PARTNER PREFERENCES</h4>            
        </div>
        """, unsafe_allow_html=True)

        # Modern radio button styling
        st.markdown("""
        <style>
        .stRadio .stSlider [role=radiogroup] {
            background: rgba(15,12,41,0.7);
            border: 1px solid #D4AF37;
            border-radius: 8px;
            padding: 15px;
        }
        .stRadio .stSlider label {
            color: #F5DEB3 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size: 22px; color: #F5DEB3; font-weight: bold; margin-bottom:-40px; margin-top:20px;">您的性别 / Your Gender</p>',
                    unsafe_allow_html=True)
        gender = st.radio("", ["Male", "Female"], horizontal=True)
        st.markdown(
            '<p style="font-size: 22px; color: #F5DEB3; font-weight: bold; margin-bottom:-40px; margin-top:20px;">理想伴侣性别 / Preferred Partner Gender</p>',
            unsafe_allow_html=True)
        partner_gender = st.radio("", ["Male", "Female", "No Preference"], index=None, key="partner_gender")

        age_range = st.slider("理想伴侣年龄 / Preferred Partner Age Range", 18, 90, (25, 45))

    # Divider with tech-inspired design
    st.markdown("""
    <div style="text-align: center; 
                margin: 40px 0;
                position: relative;">
        <div style="height: 1px; 
                   background: linear-gradient(90deg, transparent 0%, #D4AF37 50%, transparent 100%);
                   width: 100%;"></div>
        <div style="position: absolute;
                   top: 50%;
                   left: 50%;
                   transform: translate(-50%, -50%);
                   background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
                   padding: 0 20px;
                   border: 1px solid #D4AF37;
                   border-radius: 20px;">
            <span style="color: #D4AF37;
                         font-size: 18px;">✦ ❈ ✦</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Analysis Button with high-tech styling
    analyze_clicked = st.button(
        "面相分析 ANALYZE MY FACE",
        type="primary",
        use_container_width=True,
        help="Click to analyze your facial features and find your ideal partner",
        key="analyze_button"
    )

    # Handle Analysis
    if uploaded_file and analyze_clicked:
        if not partner_gender:
            st.error("请先选择伴侣性别 / Please select a partner gender first")
            st.stop()

        with st.spinner("分析中，请稍候..."):
            show_loading_animation()
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                # Convert age range tuple to string
                age_range_str = f"{age_range[0]}-{age_range[1]}"

                # Normalize gender strings
                gender = gender.lower()
                partner_gender = partner_gender.lower()
                if partner_gender == "no preference":
                    partner_gender = "other"

                data = {
                    "gender": gender,
                    "partner_gender": partner_gender,
                    "age_range": age_range_str,
                    "personality": "Not Specified"
                }

                response = requests.post(API_ENDPOINT, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()

                    st.success("分析完成！请选择标签查看详细信息")

                    # TABS
                    tab1, tab2, tab3 = st.tabs([
                        "📜 **命理分析 / Destiny Analysis**",
                        "📆 **生命预测 / Life Predictions**",
                        "💞 **理想伴侣 / Ideal Partner**",
                    ])

                    with tab1:
                        display_element_section(result["measurements"])
                        display_professional_analysis(result["professional_analysis"])

                    with tab2:
                        display_life_predictions(result["predictions"])

                    with tab3:
                        display_partner_tab(result)

                else:
                    st.error(f"分析失败: {response.status_code}")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
