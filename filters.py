
import numpy as np
from PIL import Image
import statistics
import math
from scipy.ndimage import convolve
from scipy.fft import dctn, idctn 

def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel."""
    kernel = [[0] * size for _ in range(size)]
    sum_val = 0
    offset = size // 2
    
    for x in range(size):
        for y in range(size):
            exp_part = math.exp(-((x - offset) ** 2 + (y - offset) ** 2) / (2 * sigma ** 2))
            kernel[x][y] = exp_part / (2 * math.pi * sigma ** 2)
            sum_val += kernel[x][y]
    
    # Normalize the kernel
    for x in range(size):
        for y in range(size):
            kernel[x][y] /= sum_val
    
    return kernel

def gaussian_filter(image, filter_size=3, sigma=1):
    """Apply a Gaussian filter for smoothing an image."""
    img = Image.fromarray(image)
    pixels = img.load()
    width, height = img.size
    
    filtered_image = Image.new("RGB", (width, height))
    new_pixels = filtered_image.load()
    kernel = gaussian_kernel(filter_size, sigma)
    offset = filter_size // 2
    
    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            r_sum, g_sum, b_sum = 0, 0, 0
            
            for i in range(filter_size):
                for j in range(filter_size):
                    r, g, b = pixels[x + i - offset, y + j - offset]
                    weight = kernel[i][j]
                    r_sum += r * weight
                    g_sum += g * weight
                    b_sum += b * weight
            
            new_pixels[x, y] = (int(r_sum), int(g_sum), int(b_sum))
    
    return np.array(filtered_image)





def butterworth_lowpass_filter(image, cutoff_freq, n=8):
    """Apply a Butterworth low-pass filter in the frequency domain to a color image."""
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[2]):
        fft = np.fft.fft2(image[:, :, channel])
        fft_shifted = np.fft.fftshift(fft)

        rows, cols = image.shape[:2]
        mask = np.zeros((rows, cols))
        center = (rows // 2, cols // 2)

        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                mask[i, j] = 1 / (1 + (D / cutoff_freq) ** (2 * n))

        filtered = fft_shifted * mask
        fft_inverse = np.fft.ifftshift(filtered)
        filtered_image[:, :, channel] = np.fft.ifft2(fft_inverse).real

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def anisotropic_diffusion(img, iterations=20, kappa=30, gamma=0.20, option=1):
    """Apply anisotropic diffusion for noise removal."""
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    padded = np.pad(img, ((1,1), (1,1), (0,0)), mode='reflect')

    for _ in range(iterations):
        delta_n = padded[:-2, 1:-1] - padded[1:-1, 1:-1]
        delta_s = padded[2:, 1:-1] - padded[1:-1, 1:-1]
        delta_e = padded[1:-1, 2:] - padded[1:-1, 1:-1]
        delta_w = padded[1:-1, :-2] - padded[1:-1, 1:-1]

        if option == 1:
            c_n = np.exp(-(delta_n/kappa)**2)
            c_s = np.exp(-(delta_s/kappa)**2)
            c_e = np.exp(-(delta_e/kappa)**2)
            c_w = np.exp(-(delta_w/kappa)**2)
        else:
            c_n = 1 / (1 + (delta_n/kappa)**2)
            c_s = 1 / (1 + (delta_s/kappa)**2)
            c_e = 1 / (1 + (delta_e/kappa)**2)
            c_w = 1 / (1 + (delta_w/kappa)**2)

        out = img + gamma * (c_n * delta_n + c_s * delta_s + c_e * delta_e + c_w * delta_w)

        img = out.copy()
        padded = np.pad(img, ((1,1), (1,1), (0,0)), mode='reflect')

    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(image, filter_size=5):
    """Apply a Median Filter for noise removal."""
    img = Image.fromarray(image)
    pixels = img.load()
    width, height = img.size

    filtered_image = Image.new("RGB", (width, height))
    new_pixels = filtered_image.load()

    offset = filter_size // 2

    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            r_vals, g_vals, b_vals = [], [], []

            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    r, g, b = pixels[x + i, y + j]
                    r_vals.append(r)
                    g_vals.append(g)
                    b_vals.append(b)

            median_r = statistics.median(r_vals)
            median_g = statistics.median(g_vals)
            median_b = statistics.median(b_vals)

            new_pixels[x, y] = (int(median_r), int(median_g), int(median_b))

    return np.array(filtered_image)


def gaussian(x, sigma):
    return np.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateral_filter_color(image, d, sigma_s, sigma_r):
    """Bilateral filter for RGB images."""
    image = image.astype(np.float32)
    filtered_image = np.zeros_like(image)

    rows, cols, channels = image.shape
    half_d = d // 2

    # Precompute spatial Gaussian
    spatial_kernel = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            x, y = i - half_d, j - half_d
            spatial_kernel[i, j] = gaussian(np.sqrt(x**2 + y**2), sigma_s)

    for i in range(rows):
        for j in range(cols):
            W_p = 0
            filtered_pixel = np.zeros(3)

            for k in range(-half_d, half_d + 1):
                for l in range(-half_d, half_d + 1):
                    x = i + k
                    y = j + l

                    if 0 <= x < rows and 0 <= y < cols:
                        diff = image[x, y] - image[i, j]
                        diff_norm = np.linalg.norm(diff)

                        range_weight = gaussian(diff_norm, sigma_r)
                        weight = spatial_kernel[k + half_d, l + half_d] * range_weight

                        filtered_pixel += weight * image[x, y]
                        W_p += weight

            if W_p != 0:
                filtered_image[i, j] = filtered_pixel / W_p
            else:
                filtered_image[i, j] = image[i, j]

    return np.clip(filtered_image, 0, 255).astype(np.uint8)



def mean_filter(image_array, kernel_size=7):
    """Apply a mean filter for denoising using convolution."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    filtered_image = np.zeros_like(image_array, dtype=float)
    
    for c in range(image_array.shape[2]):
        filtered_image[..., c] = convolve(image_array[..., c].astype(float), kernel, mode='reflect')
    
    return np.clip(filtered_image, 0, 255).astype(np.uint8)



# === High-Pass Filter ===
def high_pass_filter_frequency(image_array, cutoff=0.1):

    def filter_channel(image_channel):
        f_transform = np.fft.fft2(image_channel)
        f_shifted = np.fft.fftshift(f_transform)
        rows, cols = image_channel.shape
    
        # Ensure r and c match the image shape
        r = np.fft.fftfreq(rows).reshape(-1, 1)  # Column vector
        c = np.fft.fftfreq(cols).reshape(1, -1)  # Row vector
    
        mask = np.sqrt(r**2 + c**2) > cutoff  # Ensure correct shape
        f_shifted *= mask  # Apply mask correctly
    
        f_ishifted = np.fft.ifftshift(f_shifted)
        filtered_channel = np.abs(np.fft.ifft2(f_ishifted))
    
        return np.clip(filtered_channel, 0, 255).astype(np.uint8)

    
    if len(image_array.shape) == 3:  # RGB Image
        return np.stack([filter_channel(image_array[:, :, i]) for i in range(3)], axis=-1)
    else:  # Grayscale Image
        return filter_channel(image_array)
