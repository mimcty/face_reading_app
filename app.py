# app.py
import streamlit as st
import requests

API_ENDPOINT = "http://localhost:8000/analyze"  # Or your deployed endpoint


def main():
    st.title("面相匹配 - Chinese Face Reading Matchmaker")

    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses Chinese face reading principles (面相学) to analyze your facial features and suggest compatible partners.")

    uploaded_file = st.file_uploader("Upload your face photo", type=["jpg", "png", "jpeg"])
    sexuality = st.selectbox(
        "Your sexuality",
        ["Heterosexual", "Homosexual", "Bisexual", "Pansexual", "Other"]
    )
    personality_types = [
        "INTJ", "INTP", "ENTJ", "ENTP", "INFJ",
        "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ",
        "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    ideal_personality = st.selectbox(
        "Preferred partner personality (optional)",
        personality_types
    )

    if uploaded_file and st.button("Analyze My Face"):
        with st.spinner("Analyzing your facial features..."):
            try:
                # Get file bytes directly - no temp file needed
                file_bytes = uploaded_file.getvalue()

                # Call API with file bytes
                files = {"file": (uploaded_file.name, file_bytes, f"image/{uploaded_file.type.split('/')[1]}")}
                data = {
                    "sexuality": sexuality.lower(),  # Ensure lowercase to match backend expectations
                    "personality": ideal_personality if ideal_personality else "INTJ"  # Default value
                }

                response = requests.post(API_ENDPOINT, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()

                    st.success("Analysis Complete!")

                    # Display scientific analysis if available
                    if "scientific_analysis" in result:
                        with st.expander("Scientific Analysis"):
                            st.json(result["scientific_analysis"])

                    # Display Chinese reading if available
                    if "chinese_reading" in result:
                        with st.expander("Chinese Face Reading"):
                            for section, content in result["chinese_reading"].items():
                                st.subheader(section.capitalize())
                                st.write(content)

                    # Display compatibility if available
                    if "compatibility" in result:
                        with st.expander("Compatibility Analysis"):
                            st.write(result["compatibility"])

                    # Display the image if available
                    if "image_url" in result:
                        st.image(
                            API_ENDPOINT.rsplit('/', 1)[0] + result["image_url"],
                            caption="Your Ideal Partner",
                            width=300
                        )

                    # Also display the original uploaded image
                    st.image(uploaded_file, caption="Your Face", width=300)
                else:
                    st.error(f"Analysis failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


if __name__ == "__main__":
    main()