import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt

from inference import predict_uploaded_image

st.set_page_config(
    page_title="Image Caption Generator",
    layout="wide"
)

st.title("Image Caption Generator with Attention")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

beam_size = st.sidebar.slider(
    "Beam Size",
    min_value = 1,
    max_value = 10,
    value = 5
)

if uploaded_file is not None:


    caption, display_image, plot_words, plot_attention = predict_uploaded_image(
        uploaded_file,
        beam_search=True,
        beam_size=beam_size,
        show_attention=True
    )

    # Show original uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.subheader("Generated Caption")
    st.success(caption)

    st.subheader("Attention Maps")

    num_cols = 3

    for row_start in range(0, len(plot_words), num_cols):

        cols = st.columns(num_cols)

        for col_idx in range(num_cols):

            idx = row_start + col_idx

            if idx >= len(plot_words):
                break

            word = plot_words[idx]
            alpha = plot_attention[idx]

            alpha = alpha.reshape(7, 7).unsqueeze(0).unsqueeze(0)

            alpha = F.interpolate(
                alpha,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            alpha = alpha.squeeze().cpu().numpy()

            fig, ax = plt.subplots(figsize=(3, 3))

            ax.imshow(display_image)
            ax.imshow(alpha, cmap="jet", alpha=0.5)

            ax.set_title(word, fontsize=12)
            ax.axis("off")

            with cols[col_idx]:
                st.pyplot(fig, use_container_width=True)

            plt.close(fig)