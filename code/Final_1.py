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

# Homomorphic Filtering 함수
def homomorphic_filtering(img, sigma=10, gamma1=0.3, gamma2=1.5):
    rows, cols = img.shape

    imgLog = np.log1p(np.array(img, dtype='float') / 255) # Log 변환

    # 이미지 크기 확장
    M = 2 * rows + 1
    N = 2 * cols + 1

    # Gaussian 저주파 필터 및 고주파 필터 생성
    X, Y = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    Xc, Yc = np.ceil(N / 2), np.ceil(M / 2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2

    LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma)) # 저주파 필터
    HPF = 1 - LPF

    LPF_shift = np.fft.ifftshift(LPF)  # LPF 필터를 중심으로 이동
    HPF_shift = np.fft.ifftshift(HPF)

    # FFT + 필터링
    img_FFT = np.fft.fft2(imgLog, (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT * LPF_shift, (M, N))) # 저주파 성분
    img_HF = np.real(np.fft.ifft2(img_FFT * HPF_shift, (M, N)))

    # 조명/반사 성분 가중치 적용
    img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

    # 역 로그 변환
    img_exp = np.expm1(img_adjusting)

    # 정규화 및 unit8 변환
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255 * img_exp, dtype='uint8')

    return img_out

# 노이즈 제거 filtering
def preprocess_barcode_image(gray, kernel_size=3):
    homo_img = homomorphic_filtering(gray)
    binary = global_threshold(homo_img)
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

    gray = cv2.imread("barcode1.jpg", cv2.IMREAD_GRAYSCALE)

    # noise 제거 전처리 진행 먼저함
    cleaned = preprocess_barcode_image(gray, kernel_size=3)

    # 바코드 detect
    result, _, vertical_map, closed_map = detect_barcode(gray, cleaned, line_thresh=180, morph_kernel_size=(5,30), area_thresh=500)

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Original Gray")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Cleaned Binary")
    plt.imshow(cleaned, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Vertical Response")
    plt.imshow(vertical_map, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Detected Barcode")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
