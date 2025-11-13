import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "Pic.jpg"
file_path = os.path.join(".", FILE_NAME)


def rms_contrast(img: np.ndarray) -> float:
    """RMS contrast (std-dev of grayscale intensities in [0,1])."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    g = gray.astype(np.float32) / 255.0
    return float(g.std())

def show_rgb_hsv_channels_gray(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                               h: np.ndarray, s: np.ndarray, v: np.ndarray) -> None:
    if not (r.shape == g.shape == b.shape == h.shape == s.shape == v.shape):
        raise ValueError("All channels must have the same dimensions (H, W).")

    def _plot_rgb_gray(r, g, b):
        plt.subplot(2, 3, 1)
        plt.imshow(r, cmap='gray')
        plt.title("R (Red Channel)")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(g, cmap='gray')
        plt.title("G (Green Channel)")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(b, cmap='gray')
        plt.title("B (Blue Channel)")
        plt.axis("off")

    def _plot_hsv_gray(h, s, v):
        plt.subplot(2, 3, 4)
        plt.imshow(h, cmap='gray')
        plt.title("H (Hue Channel)")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(s, cmap='gray')
        plt.title("S (Saturation Channel)")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(v, cmap='gray')
        plt.title("V (Value Channel)")
        plt.axis("off")

    plt.figure(figsize=(12, 8))
    _plot_rgb_gray(r, g, b)
    _plot_hsv_gray(h, s, v)
    plt.tight_layout()
    plt.show()

 
def show_rgb_channels_colored(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> None:
    if r.shape != g.shape or g.shape != b.shape:
        raise ValueError("All channels must have the same dimensions (H, W).")

    red_img = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2)
    green_img = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=2)
    blue_img = np.stack([np.zeros_like(b), np.zeros_like(b), b], axis=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(red_img.astype(np.uint8))
    plt.title("R (Red Channel)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(green_img.astype(np.uint8))
    plt.title("G (Green Channel)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blue_img.astype(np.uint8))
    plt.title("B (Blue Channel)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_hsv_channels_colored(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> None:
    if h.shape != s.shape or s.shape != v.shape:
        raise ValueError("All channels must have the same dimensions (H, W).")

    h_colored = cv2.cvtColor(
        cv2.merge([h, np.full_like(s, 255), np.full_like(v, 255)]),
        cv2.COLOR_HSV2RGB
    )
    s_colored = cv2.cvtColor(
        cv2.merge([np.full_like(h, 90), s, np.full_like(v, 255)]),
        cv2.COLOR_HSV2RGB
    )
    v_colored = cv2.cvtColor(
        cv2.merge([np.full_like(h, 90), np.full_like(s, 255), v]),
        cv2.COLOR_HSV2RGB
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(h_colored)
    plt.title("H (Hue Channel)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(s_colored)
    plt.title("S (Saturation Channel)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(v_colored)
    plt.title("V (Value Channel)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    # Part 1
    img = cv2.imread(file_path)
    
    if img is None:
        raise FileNotFoundError("Sth is wrong with the file. It was not found.")

    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for myself
    print(img)
    print(type(img))

    print('Image data type:', img.dtype)
    print("Image shape: ", img.shape)

    if len(img.shape) == 2:  # (height, width)
        print("This is a grayscale image (1 channel).")
    elif len(img.shape) == 3:  # (height, width, channels)  channels -> BGR
        channels = img.shape[2]
        print(f"This is a color image with {channels} channels.")
    else:
        print("Unknown image format.")




    # Part 2 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Display the RGB image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("RGB Image")
    plt.axis('off')

    # Display the Grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    # Display the Binary image
    plt.subplot(1, 3, 3)
    plt.imshow(img_binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')

    plt.show()




    # Part 3
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Display the HSV image
    plt.subplot(1, 3, 3)
    plt.imshow(img_hsv)
    plt.title("HSV Image")
    plt.axis('off')

    plt.show()





    # Part 4
    r, g, b = cv2.split(img_rgb)
    h, s, v = cv2.split(img_hsv)

    # --- RGB --- --- HSV ---
    show_rgb_hsv_channels_gray(r, g, b, h, s, v)

    # Show channels (colored)
    show_rgb_channels_colored(r, g, b)
    show_hsv_channels_colored(h, s, v)





    # Part 5
    contrast = img_gray.std()

    print("Contrast base image:", contrast)

    img_higher = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    img_lower = cv2.convertScaleAbs(img, alpha=0.6, beta=0)

    contrast_higher = cv2.cvtColor(img_higher, cv2.COLOR_BGR2GRAY).std()
    contrast_lower = cv2.cvtColor(img_lower, cv2.COLOR_BGR2GRAY).std()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original\nContrast={contrast:.4f}")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(img_higher, cv2.COLOR_BGR2RGB))
    plt.title(f"Higher Contrast\nContrast={contrast_higher:.4f}")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(img_lower, cv2.COLOR_BGR2RGB))
    plt.title(f"Lower Contrast\nContrast={contrast_lower:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    




    # Part 6
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(6, 4))
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    plt.title("Color Histogram (BGR Channels)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()







    # Part 7

    # Histogram Stretching
    img_stretched = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Histogram Equalization
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y_eq = cv2.equalizeHist(Y)
    img_equalized = cv2.cvtColor(cv2.merge([Y_eq, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

    # for display
    stretched_rgb = cv2.cvtColor(img_stretched, cv2.COLOR_BGR2RGB)
    equalized_rgb = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB)


    # Show all
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(stretched_rgb)
    plt.title("Histogram Stretched")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(equalized_rgb)
    plt.title("Histogram Equalized")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.hist(img_rgb.ravel(), bins=256, range=(0,256))
    plt.title("Original Histogram")

    plt.subplot(2, 3, 5)
    plt.hist(stretched_rgb.ravel(), bins=256, range=(0,256))
    plt.title("Stretched Histogram")

    plt.subplot(2, 3, 6)
    plt.hist(equalized_rgb.ravel(), bins=256, range=(0,256))
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()
