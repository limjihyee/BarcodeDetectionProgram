# Barcode Detection in Noisy Environments

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [데이터 및 노이즈 유형](#데이터-및-노이즈-유형)  
3. [알고리즘 상세 설명](#알고리즘-상세-설명)  
   - 3.1 [공통 전처리 파이프라인](#공통-전처리-파이프라인)  
   - 3.2 [Barcode1: 조명 불균형](#barcode1-조명-불균형)  
   - 3.3 [Barcode2: Salt-and-Pepper 노이즈](#barcode2-salt-and-pepper-노이즈)  
   - 3.4 [Barcode3: 대각선 격자 노이즈](#barcode3-대각선-격자-노이즈)  
4. [결과](#결과)  
---

## 프로젝트 개요  
다양한 노이즈(조명 불균형, Salt-and-Pepper, 45°/–45° 대각선 격자)가 포함된 영상에서 바코드를 안정적으로 검출하는 파이프라인을 설계 및 구현함
- **목표**: 각 노이즈 환경별로 최적화된 전처리 및 필터링 기법을 적용하고, 동일한 검출 모듈을 통해 바코드를 정확히 인식 (ML, DL은 허용하지 않음)

## 데이터 및 노이즈 유형  
- **Barcode1**  
  - 특징: 화면 상·하단 조명 강도 차이  
- **Barcode2**  
  - 특징: Salt-and-Pepper 노이즈가 랜덤 분포  
- **Barcode3**  
  - 특징: ±45° 방향의 grid 노이즈가 바코드와 겹침

## 알고리즘 상세 설명
- 공통 전처리 파이프라인
   1. 그레이스케일 변환: 컬러 영상 → 단일 채널
   2. 노이즈별 전처리: 각 스크립트(final1.py, final2.py, final3.py)에서 구체적 처리
   3. 바코드 검출
      - 수직 에지 강조: 수직 커널 필터를 적용해 세로선만 검출되도록 함.
      - 이진화: binary thresholding 진행 
      - Closing으로 묶어서 하나의 덩어리로 판단 → Contour 검출 → 최소 외접 사각형으로 Bounding Box 추출
- Barcode1: 조명 불균형
  1. Homomorphic Filtering
    - Homomorphic filtering을 진행하여 직접적인 광원으로부터의 빛 즉, low frequency components를 약화시키고 물체로부터 반사된 빛 즉 high frequency components를 강화시켜 이미지의 details을 강화시킴
    - 로그 변환 → DFT → H(U,V) → IDFT → 지수 변환
  2. Global Thresholding
  3. Morphology 연산
    - 이진화 후 Closing → Opening 진행
  4. 검출
    - vertical edge → threshold → contour
- Barcode2: Salt-and-Pepper 노이즈
   1. Global Threshoing
      - 초기값 T=127 설정 → 픽셀 두 그룹 평균으로 thresholding을 업데이트하는 방법을 사용 : 배경과 바코드 확실히 분리시킴 (애매한 점들은 제거될 수 있도록 함)
   2. Closing → Opening
      - closing을 진행하여 바코드 막대 사이의 작은 틈을 매울 수 있도록 하였고 제거되지 않은 점(즉 구멍)들을 채울 수 있도록 함. opening을 통해서 closing에 의해 끊긴 바코드를 검정색으로 다시 연결함.
   4. 검출: 공통 모듈 사용
- Barcode3: 대각선 격자 노이즈
   1. Line Detection Kernel
      - 45°/–45° 커널 이용해 필터링 → 대각선 성분 차감
   2. 검출: 공통 모듈 사용
 
  ## 결과
  Barcode1 : 조명 불균형은 제거했지만, Closing 커널 폭 미세조정 필요 따라서 Gaussian Low Pass Filter 적용하는 등 시도 진행하면 좋을 듯.
  Barcode2 : 소금-후추 노이즈 대부분 제거됨. 조금 더 간단하게 진행하고자 한다면 Median Filter를 적용해서 진행
  Barcode3 : 대각선 노이즈 완벽히 제거가 안되고 되려 뭉게짐 따라서 Fourier 변환을 진행해서 공간 domain(이미지 영역)을 주파수 domain으로 변환 시켜 노이즈가 뚜렷한 특정 주파수 위치를 파악하고
             대각선 격자 노이즈가 위치한 주파수 영역만 골라 차단 진행한다. 이후 약하게 GLPF 적용시켜 잔여 노이즈 제거 진행한 뒤, Inverse FFT하여 필터링 완료된 주파수 성분을 다시 공간 영역으로 하여 이미지 결과 얻는 식으              로 하면 좋을 듯. (주파수 domian에서 노이즈 영역 확인 후 masking 진행하려고 했으나 생각보다 잘 안나와서 추후 고민 해봐야 할 듯.)

  
