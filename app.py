import streamlit as st
import numpy as np
import imageio
import torch
from filters import butterworth_lowpass_filter, anisotropic_diffusion, median_filter, bilateral_filter_color, gaussian_filter, mean_filter, bm3d_denoise_poisson, high_pass_filter_frequency
from gan_model import load_gan_model, preprocess_image, denoise_image as gan_denoise_image

# Set up Streamlit UI
st.title("Denoising the Images with Deep Learning & Traditional Filters")

uploaded_file = st.file_uploader("Upload a Noisy Image", type=["png", "jpg", "jpeg"])

# Load Deep Learning Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GAN model
gan_model_path = "generator_model.pth"
try:
    gan_model = load_gan_model(gan_model_path)
    st.success("GAN Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading GAN model: {e}")


if uploaded_file:
    noisy_image = imageio.imread(uploaded_file)

    if noisy_image.ndim == 2:
        noisy_image = np.stack([noisy_image] * 3, axis=-1)

    st.image(noisy_image, caption="Noisy Image", use_column_width=True)

    denoise_choice = st.selectbox(
        "Choose a Denoising Method",
        [
            "Butterworth Low-Pass",
            "Anisotropic Diffusion",
            "Median Filter",
            "Bilateral Filter",
            "Gaussian Filter",
            "Mean Filter",
            "BM3D Poisson Noise Reduction",
            "High-Pass Filter",
            "GAN-Based Denoising"
           
        ],
    )

    if denoise_choice == "Butterworth Low-Pass":
        cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=200, value=60)
        if st.button("Denoise Image"):
            denoised_image = butterworth_lowpass_filter(noisy_image, cutoff)

    elif denoise_choice == "Anisotropic Diffusion":
        iterations = st.slider("Iterations", min_value=5, max_value=50, value=20)
        kappa = st.slider("Kappa", min_value=10, max_value=100, value=30)
        gamma = st.slider("Gamma", min_value=0.05, max_value=0.5, value=0.2)
        option = st.radio("Diffusion Function", [1, 2])
        if st.button("Denoise Image"):
            denoised_image = anisotropic_diffusion(noisy_image, iterations, kappa, gamma, option)

    elif denoise_choice == "Median Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=5, step=2)
        if st.button("Denoise Image"):
            denoised_image = median_filter(noisy_image, filter_size)

    elif denoise_choice == "Bilateral Filter":
        d = st.slider("Filter Window Size", min_value=3, max_value=15, value=9, step=2)
        sigma_s = st.slider("Spatial Sigma", min_value=1, max_value=50, value=10)
        sigma_r = st.slider("Range Sigma", min_value=1, max_value=100, value=25)
        if st.button("Denoise Image"):
            denoised_image = bilateral_filter_color(noisy_image, d, sigma_s, sigma_r)

    elif denoise_choice == "Gaussian Filter":
        filter_size = st.slider("Filter Size", min_value=3, max_value=15, value=3, step=2)
        sigma = st.slider("Sigma Value", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        if st.button("Denoise Image"):
            denoised_image = gaussian_filter(noisy_image, filter_size, sigma)

    elif denoise_choice == "Mean Filter":
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=7, step=2)
        if st.button("Denoise Image"):
            denoised_image = mean_filter(noisy_image, kernel_size)

    elif filter_choice == "BM3D Poisson Noise Reduction":
        sigma = st.slider("Noise Level (σ)", min_value=0.05, max_value=1.0, value=0.1)
        if st.button("Apply Filter"):
            filtered_image = bm3d_denoise_poisson(image, sigma)

    elif filter_choice == "High-Pass Filter":
        cutoff = st.slider("Cutoff Frequency", min_value=0.1, max_value=1.0, value=0.4)
        if st.button("Apply Filter"):
            filtered_image = high_pass_filter_frequency(image, cutoff)

    elif denoise_choice == "GAN-Based Denoising":
        if st.button("Denoise Image"):
            try:
                noisy_tensor = preprocess_image(uploaded_file)
                denoised_image = gan_denoise_image(gan_model, noisy_tensor)
            except Exception as e:
                st.error(f"Error: {e}")
                denoised_image = noisy_image


    # Display results
    if "filtered_image" in locals():
        st.subheader("Denoising Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(noisy_image, caption="Noisy Image", use_column_width=True)
        with col2:
            st.image(filtered_image, caption="Denoised Image", use_column_width=True)
        
        # Calculate and display metrics comparing denoised with original
        st.subheader("Denoising Quality Metrics")
        
        if original_image.shape == filtered_image.shape:
            try:
                # PSNR calculation
                denoised_psnr = psnr(original_image, filtered_image, data_range=255)
                
                # SSIM calculation (convert to grayscale if needed)
                original_gray = np.mean(original_image, axis=2) if original_image.ndim == 3 else original_image
                filtered_gray = np.mean(filtered_image, axis=2) if filtered_image.ndim == 3 else filtered_image
                denoised_ssim = ssim(original_gray, filtered_gray, data_range=255)
                
                st.write(f"PSNR (Original vs Denoised): {denoised_psnr:.2f} dB")
                st.write(f"SSIM (Original vs Denoised): {denoised_ssim:.4f}")
                
                # Show improvement over noisy image
                if 'noisy_psnr' in locals() and 'noisy_ssim' in locals():
                    st.write(f"PSNR Improvement: {denoised_psnr - noisy_psnr:.2f} dB")
                    st.write(f"SSIM Improvement: {denoised_ssim - noisy_ssim:.4f}")
                
                # Interpretation
                st.caption("Higher PSNR values indicate better quality (typically >30 dB is good).")
                st.caption("SSIM ranges from -1 to 1, with 1 being perfect similarity.")
            except ValueError as e:
                st.warning(f"Could not calculate metrics: {e}")
        else:
            st.warning("Original and denoised images must have the same dimensions for comparison")
