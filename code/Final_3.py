import numpy as np
import cv2
import matplotlib.pyplot as plt

# Line‐detection kernels
sobel_x_kernel = np.array([[-2, -1,  0],
                           [-1,  0,  1],
                           [ 0,  1,  2]])

sobel_y_kernel = np.array([[ 0,  1,  2],
                           [-1,  0,  1],
                           [-2, -1,  0]])

vertical_kernel = np.array([[-1, 2, -1],
                            [-1, 2, -1],
                            [-1, 2, -1]])

# convolution function
def convolve(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant', constant_values=0)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kernel)
    return out

# Morphological Operations
def erosion(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), constant_values=255)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            reg = padded[i:i+kh, j:j+kw]
            if np.all(reg[kernel==1] == 255):
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
            reg = padded[i:i+kh, j:j+kw]
            if np.any(reg[kernel==1] == 255):
                out[i,j] = 255
    return out

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# global_thresholding
def global_threshold(img):
    T, prev = 127, 0
    while T != prev:
        prev = T
        G1 = img[img > T]
        G2 = img[img <= T]
        m1 = G1.mean() if G1.size>0 else 0
        m2 = G2.mean() if G2.size>0 else 0
        T = int((m1 + m2)/2)
    return np.where(img > T, 255, 0).astype(np.uint8)

# 대각선 Sobel를 통해서 noise 제거 filtering
def preprocess_barcode_image(gray, kernel_size=3):
    # 대각선 Sobel로 대각 노이즈 검출
    img_f = gray.astype(np.float32)
    sx = convolve(img_f, sobel_x_kernel)
    sy = convolve(img_f, sobel_y_kernel)
    abs_sx = np.abs(sx)
    abs_sy = np.abs(sy)

    # I - |Sx| - |Sy|를 해서 각 성분에서의 노이즈를 제거할 수 있도록 함
    filtered = img_f - abs_sx - abs_sy
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    cleaned = filtered

    return cleaned

# barcode 검출
def detect_barcode(gray, cleaned, line_thresh=180, morph_kernel_size=(5,30), area_thresh=500):
    # 세로선만 검출
    vertical = cv2.filter2D(cleaned, -1, vertical_kernel)

    # 이진화
    _, vert_bin = cv2.threshold(vertical, line_thresh, 255, cv2.THRESH_BINARY)

    # closing으로 띠 연결 묶기
    mk = np.ones(morph_kernel_size, dtype=np.uint8)
    closed = closing(vert_bin, mk)

    # 컨투어 -> 최대 영역 박스로
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contours:
        good = [c for c in contours if cv2.contourArea(c) > area_thresh]
        if good:
            x,y,w,h = cv2.boundingRect(max(good, key=cv2.contourArea))
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 3)
    return out, cleaned, vertical, closed


if __name__ == "__main__":
    gray = cv2.imread("barcode3.jpg", cv2.IMREAD_GRAYSCALE)

    # filtering
    cleaned, = preprocess_barcode_image(gray, kernel_size=3),

    # 바코드 검출 진행
    result, _, vertical_map, closed_map = detect_barcode(gray, cleaned, line_thresh=180, morph_kernel_size=(5,30), area_thresh=500)

    # 시각화
    plt.figure(figsize=(12, 8))
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
