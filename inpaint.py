import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imsave, imread
import os
from tqdm import tqdm

image_path = "./img/0.jpeg"
image = img_as_float(imread(image_path)).copy()
h, w, _ = image.shape


mask = np.zeros((h, w), dtype=np.uint8)
drawing = False
ix, iy = -1, -1

def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mask, image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), 5, 255, -1)
        image_display = (image * 255).astype(np.uint8).copy()
        overlay = np.zeros_like(image_display)
        overlay[mask > 0] = [0, 0, 255]
        image_display = cv2.addWeighted(image_display, 0.7, overlay, 0.3, 0)
        cv2.imshow(window_name, image_display)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), 5, 255, -1)
        image_display = (image * 255).astype(np.uint8).copy()
        overlay = np.zeros_like(image_display)
        overlay[mask > 0] = [0, 0, 255]
        image_display = cv2.addWeighted(image_display, 0.7, overlay, 0.3, 0)
        cv2.imshow(window_name, image_display)

window_name = 'Draw Mask (ESC to continue)'
image_display = (image * 255).astype(np.uint8).copy()
image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_mask)
cv2.imshow(window_name, image_display)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()


mask_bool = (mask > 0)
damaged = image.copy()
damaged[mask_bool] = 0.0


def laplace_inpaint_channel(channel, mask, num_iters=1000):
    u = channel.copy()
    for _ in tqdm(range(num_iters), desc="Laplace Inpainting"):
        u_old = u.copy()
        u[mask] = 0.25 * (
            np.roll(u_old, 1, axis=0)[mask] +
            np.roll(u_old, -1, axis=0)[mask] +
            np.roll(u_old, 1, axis=1)[mask] +
            np.roll(u_old, -1, axis=1)[mask]
        )
    return u


inpainted_channels = []
for c in range(3):
    inpainted = laplace_inpaint_channel(damaged[:, :, c], mask_bool)
    inpainted_channels.append(inpainted)
inpainted_color = np.stack(inpainted_channels, axis=2)


psnr_values, ssim_values = [], []
for c in range(3):
    original_c = image[:, :, c][mask_bool]
    restored_c = inpainted_color[:, :, c][mask_bool]
    psnr = peak_signal_noise_ratio(original_c, restored_c, data_range=1.0)
    ssim = structural_similarity(original_c, restored_c, data_range=1.0)
    psnr_values.append(psnr)
    ssim_values.append(ssim)

print("Laplace PDE 修复结果：")
print(f"PSNR (R,G,B): {psnr_values[0]:.2f}, {psnr_values[1]:.2f}, {psnr_values[2]:.2f} dB")
print(f"SSIM (R,G,B): {ssim_values[0]:.4f}, {ssim_values[1]:.4f}, {ssim_values[2]:.4f}")


damaged_uint8 = (np.clip(damaged, 0, 1) * 255).astype(np.uint8)
damaged_bgr = cv2.cvtColor(damaged_uint8, cv2.COLOR_RGB2BGR)

gaussian_blur = cv2.GaussianBlur(damaged_bgr, (9, 9), 0)
gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB) / 255.0

mean_blur = cv2.blur(damaged_bgr, (9, 9))
mean_blur_rgb = cv2.cvtColor(mean_blur, cv2.COLOR_BGR2RGB) / 255.0

def evaluate(method_img, name=""):
    psnrs, ssims = [], []
    for c in range(3):
        orig = image[:, :, c][mask_bool]
        restored = method_img[:, :, c][mask_bool]
        psnr = peak_signal_noise_ratio(orig, restored, data_range=1.0)
        ssim = structural_similarity(orig, restored, data_range=1.0)
        psnrs.append(psnr)
        ssims.append(ssim)
    print(f"{name} - PSNR (R,G,B): {psnrs[0]:.2f}, {psnrs[1]:.2f}, {psnrs[2]:.2f} dB")
    print(f"{name} - SSIM (R,G,B): {ssims[0]:.4f}, {ssims[1]:.4f}, {ssims[2]:.4f}")
    return psnrs, ssims

mean_psnr, mean_ssim = evaluate(mean_blur_rgb, "Mean Filter")
gauss_psnr, gauss_ssim = evaluate(gaussian_blur_rgb, "Gaussian Filter")


os.makedirs("results", exist_ok=True)
imsave("results/original.png", (image * 255).astype(np.uint8))
imsave("results/damaged.png", (damaged * 255).astype(np.uint8))
imsave("results/inpainted_laplace.png", (inpainted_color * 255).astype(np.uint8))
imsave("results/mean_filter.png", (mean_blur_rgb * 255).astype(np.uint8))
imsave("results/gaussian_filter.png", (gaussian_blur_rgb * 255).astype(np.uint8))
imsave("results/mask.png", mask)


fig, axs = plt.subplots(1, 5, figsize=(20, 4))
axs[0].imshow(image)
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(damaged)
axs[1].set_title("Damaged")
axs[1].axis('off')

axs[2].imshow(inpainted_color)
axs[2].set_title("Laplace PDE")
axs[2].axis('off')

axs[3].imshow(mean_blur_rgb)
axs[3].set_title("Mean Filter")
axs[3].axis('off')

axs[4].imshow(gaussian_blur_rgb)
axs[4].set_title("Gaussian Filter")
axs[4].axis('off')

plt.tight_layout()
plt.savefig("results/comparison_all_methods.png", dpi=200)
plt.show()
