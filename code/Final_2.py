import numpy as np
import cv2
import matplotlib.pyplot as plt

# Line‐detection kernels
horizontal_kernel = np.array([[-1,-1,-1],
                                [ 2, 2, 2],
                                [-1,-1,-1]])
vertical_kernel = np.array([[-1, 2, -1],
                            [-1, 2, -1],
                            [-1, 2, -1]])
pos45_kernel = np.array([[-1, -1, 2],
                            [-1, 2, -1],
                            [ 2, -1, -1]])
neg45_kernel = np.array([[ 2, -1, -1],
                            [-1, 2, -1],
                            [-1, -1, 2]])

# Morphological Operations
def erosion(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), constant_values=255)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.all(region[kernel==1] == 255):
                out[i,j] = 255
    return out

def dilation(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), constant_values=0)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel==1] == 255):
                out[i,j] = 255
    return out

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# Global Thresholding
def global_threshold(img):
    T, prev_T = 127, 0
    while T != prev_T:
        prev_T = T
        G1 = img[img > T]
        G2 = img[img <= T]

        m1 = G1.mean() if G1.size>0 else 0
        m2 = G2.mean() if G2.size>0 else 0
        T = int((m1 + m2) / 2)
    return np.where(img > T, 255, 0).astype(np.uint8)

# 노이즈 제거 filtering
def remove_noise(gray, kernel_size=3):
    binary = global_threshold(gray)
    k = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = closing(binary, k)
    cleaned = opening(cleaned, k)
    return cleaned

#  Barcode Detection
def detect_barcode(gray, cleaned, line_thresh=180, morph_kernel_size=(5,30), area_thresh=500):

    # 세로선만 검출
    vertical = cv2.filter2D(cleaned, -1, vertical_kernel)

    # 이진화
    _, vert_bin = cv2.threshold(vertical, line_thresh, 255, cv2.THRESH_BINARY)

    # closing으로 띠 연결 묶기
    mk = np.ones(morph_kernel_size, dtype=np.uint8)
    closed = closing(vert_bin, mk)

    # 컨투어 -> 가장 큰 영역에 박스로
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if contours:
        good = [c for c in contours if cv2.contourArea(c) > area_thresh]
        if good:
            largest = max(good, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 3)

    return out, cleaned, vertical, closed


if __name__ == "__main__":

    gray = cv2.imread("barcode2.jpg", cv2.IMREAD_GRAYSCALE)

    # noise 제거 전처리 진행 먼저함
    cleaned = remove_noise(gray, kernel_size=3)

    # 바코드 detect
    result, _, vertical_map, closed_map = detect_barcode(gray, cleaned, line_thresh=180, morph_kernel_size=(5,30), area_thresh=500)

    # 시각화
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.title("Original Gray")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("Cleaned Binary")
    plt.imshow(cleaned, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.title("Vertical Response")
    plt.imshow(vertical_map, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("Detected Barcode")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
