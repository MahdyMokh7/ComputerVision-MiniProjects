import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "Pic.jpg"
file_path = os.path.join(".", FILE_NAME)

if __name__ == "__main__":

    # Part 1
    img = cv2.imread(filename=file_path)

    if img is None:
        raise FileNotFoundError(f"{file_path} not found or couldnt open.")
    




    # Part 2
    amount_percentage_affected = 0.04  # (4%)
    s_vs_p = 0.5  # half half split

    img_noisy = img.copy()
    rows, cols, _ = img.shape

    # Number of salt and pepper pixels
    num_salt = np.ceil(amount_percentage_affected * img.size * s_vs_p / 3)  # divide by 3 (for 3 color channels)
    num_pepper = np.ceil(amount_percentage_affected * img.size * (1.0 - s_vs_p) / 3)

    # Add Salt (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    img_noisy[coords[0], coords[1], :] = 255

    # Add Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    img_noisy[coords[0], coords[1], :] = 0


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB))
    plt.title("Salt & Pepper Noise")
    plt.axis("off")

    plt.tight_layout()
    plt.show()




    # Part 4
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # I put the keranl size bigger than usual so we could see the differnce and the effect of each filter

    # Mean Filter
    mean_filtered = cv2.blur(img, (15, 15))

    # Gaussian Filter
    gaussian_filtered = cv2.GaussianBlur(img, (15, 15), 0) 

    # Median Filter
    median_filtered = cv2.medianBlur(img, 15) 


    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB))
    plt.title("Mean Filter")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Filter")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    plt.title("Median Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    




    # Part 5 

    # because the salt and peper is a random noise so the Median filter is the best to apply to reduce and remove noise
    denoised = cv2.medianBlur(img_noisy, 5)

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB))
    plt.title("Noisy Image (Salt & Pepper)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    plt.title("After Median Filter (Noise Removed)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()  





    # Part 6 
    # Sobel Filter
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Canny Edge Detector
    edges_b = cv2.Canny(img[:, :, 0], 100, 200)
    edges_g = cv2.Canny(img[:, :, 1], 100, 200)
    edges_r = cv2.Canny(img[:, :, 2], 100, 200)
    canny_color = cv2.merge([edges_b, edges_g, edges_r])

    # Laplacian Filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(cv2.cvtColor(sobel_combined, cv2.COLOR_BGR2RGB))
    plt.title("Sobel Filter (Color)")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(cv2.cvtColor(canny_color, cv2.COLOR_BGR2RGB))
    plt.title("Canny Edge Detector (Color)")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB))
    plt.title("Laplacian Filter (Color)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # --- Sobel Filter ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
    sobel_combined = cv2.magnitude(sobelx, sobely)        # combine both directions

    # --- Canny Edge Detector ---
    canny_edges = cv2.Canny(gray, 100, 200)

    # --- Laplacian Filter ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Sobel Filter (Edges)")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(canny_edges, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(laplacian, cmap='gray')
    plt.title("Laplacian Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()