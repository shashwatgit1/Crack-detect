
import sys
from pathlib import Path

# Ensure the project root is in sys.path so 'src' can always be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from PIL import Image

from src.inference import analyze_image

st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon="🏗️",
    layout="wide"
)

st.title("Concrete Crack Detection & Severity Estimation")
st.write(
    "Upload a concrete surface image to classify visible cracks. "
    "Segmentation and visual severity estimation become available "
    "after training the U-Net model with annotated images."
)

st.warning(
    "Prototype only: predictions are not a structural safety assessment."
)

uploaded_file = st.file_uploader(
    "Upload a concrete image",
    type=["jpg", "jpeg", "png"]
    
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Analyze image", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                result = analyze_image(image)

            st.subheader("Classification result")
            st.write("**Prediction:**", result["label"])
            st.metric(
                "Crack probability",
                f"{result['crack_probability'] * 100:.2f}%"
            )
            st.metric(
                "Prediction confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            with right:
                st.subheader("Crack overlay")
                if result["mask"] is not None:
                    st.image(
                        result["overlay"],
                        caption="Red pixels indicate predicted crack regions",
                        use_container_width=True
                    )
                else:
                    st.info(
                        "No segmentation mask available. Train the U-Net "
                        "model to generate a crack overlay."
                    )

            st.subheader("Image-based severity")
            severity = result["severity"]
            st.write("**Category:**", severity["severity"])

            if "crack_area_percentage" in severity:
                st.write(
                    "**Predicted crack area:**",
                    f"{severity['crack_area_percentage']}% of the mask"
                )
                st.write(
                    "**Estimated length:**",
                    f"{severity['estimated_length_pixels']} pixels"
                )
                st.write(
                    "**Connected components:**",
                    severity["connected_components"]
                )

            st.caption(severity["note"])

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Analysis failed: {error}")
