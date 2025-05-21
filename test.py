from utils.common import *
from model import FSRCNN
import torch
import pandas as pd
import os
import torch
import torch.nn.functional as F

def SSIM(img1, img2, max_val=1):
    """구현 간단한 PyTorch용 SSIM 계산 (single image pair)"""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    img1 = img1.unsqueeze(0)  # [1, C, H, W]
    img2 = img2.unsqueeze(0)

    mu1 = F.avg_pool2d(img1, 3, 1, 0)
    mu2 = F.avg_pool2d(img2, 3, 1, 0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# ---------------------- 설정 ----------------------
scale = 3
sigma = 0.2

ckpt_path_base = "C:/codelib/git/test_FSRCNN/models/FSRCNN-x3_conv/FSRCNN-x3.pt"
ckpt_path_my   = "C:/codelib/git/test_FSRCNN/models/FSRCNN-x3_human/FSRCNN-x3.pt"
data_dir       = "C:/codelib/git/test_FSRCNN/LR_HR_maker_output_png(LR)"
label_dir      = "C:/codelib/git/test_FSRCNN/LR_HR_maker_input(HR)"
csv_output     = "C:/codelib/git/test_FSRCNN/results/psnr_ssim_comparison.csv"

# ------------------ 중심 crop 함수 ------------------
def center_crop(tensor, target_shape):
    _, H, W = tensor.shape
    th, tw = target_shape
    x1 = max((H - th) // 2, 0)
    y1 = max((W - tw) // 2, 0)
    return tensor[:, x1:x1+th, y1:y1+tw]

def crop_to_smallest(a, b):
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])
    return center_crop(a, (H, W)), center_crop(b, (H, W))

# ---------------------- main ----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    model_base = FSRCNN(scale, device)
    model_base.load_weights(ckpt_path_base)

    model_my = FSRCNN(scale, device)
    model_my.load_weights(ckpt_path_my)

    ls_data = sorted_list(data_dir)
    ls_labels = sorted_list(label_dir)

    results = []

    with torch.no_grad():
        for i in range(len(ls_data)):
            # LR & HR 이미지 불러오기
            lr_image = read_image(ls_data[i])                      # [3, h, w]
            hr_image = read_image(ls_labels[i])                    # [3, H, W]
            lr_blur = gaussian_blur(lr_image, sigma=sigma)

            # Y 채널로 변환 + 정규화
            lr_y = norm01(rgb2ycbcr(lr_blur))
            hr_y = norm01(rgb2ycbcr(hr_image))

            # Bicubic 복원
            bicubic = upscale(lr_blur, scale)
            bicubic_y = norm01(rgb2ycbcr(bicubic))

            # FSRCNN 모델 복원
            lr_input = torch.unsqueeze(lr_y, dim=0).to(device)
            sr_base = model_base.predict(lr_input)[0].cpu()
            sr_my   = model_my.predict(lr_input)[0].cpu()

            # 크기 맞추기 (가장 작은 크기 기준 crop)
            bicubic_y, hr_crop = crop_to_smallest(bicubic_y, hr_y)
            sr_base, _ = crop_to_smallest(sr_base, hr_crop)
            sr_my,   _ = crop_to_smallest(sr_my,   hr_crop)

            # PSNR 계산
            psnr_bicubic = PSNR(hr_crop, bicubic_y, max_val=1).item()
            psnr_base    = PSNR(hr_crop, sr_base,    max_val=1).item()
            psnr_my      = PSNR(hr_crop, sr_my,      max_val=1).item()

            # SSIM 계산
            ssim_bicubic = SSIM(hr_crop, bicubic_y, max_val=1).item()
            ssim_base    = SSIM(hr_crop, sr_base,    max_val=1).item()
            ssim_my      = SSIM(hr_crop, sr_my,      max_val=1).item()

            # 결과 저장
            results.append({
                "Image": os.path.basename(ls_data[i]),
                "PSNR_Bicubic": psnr_bicubic,
                "PSNR_FSRCNN": psnr_base,
                "PSNR_MyFSRCNN": psnr_my,
                "SSIM_Bicubic": ssim_bicubic,
                "SSIM_FSRCNN": ssim_base,
                "SSIM_MyFSRCNN": ssim_my
            })

            print(f"[{i+1}/{len(ls_data)}] PSNR(B): {psnr_bicubic:.2f}, F: {psnr_base:.2f}, M: {psnr_my:.2f} | "
                  f"SSIM(B): {ssim_bicubic:.4f}, F: {ssim_base:.4f}, M: {ssim_my:.4f}")

    # 평균 추가
    df = pd.DataFrame(results)
    avg = df.mean(numeric_only=True)
    avg["Image"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    # CSV 저장
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    df.to_csv(csv_output, index=False, float_format="%.4f")
    print(f"\n✅ CSV 저장 완료: {csv_output}")

if __name__ == "__main__":
    main()
