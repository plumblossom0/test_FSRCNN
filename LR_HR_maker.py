import os
from utils.common import read_image, write_image, gaussian_blur, resize_bicubic, sorted_list

# ---------------------------
# 설정
# ---------------------------
scale = 3
sigma = 0.7  # FSRCNN 논문 방식
input_dir = "C:/codelib/git/test_FSRCNN/LR_HR_maker_input(HR)"
output_dir = "C:/codelib/git/test_FSRCNN/LR_HR_maker_output_png(LR)"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# HR 이미지 다운샘플링 후 PNG 저장
# ---------------------------
hr_list = sorted_list(input_dir)

for i, hr_path in enumerate(hr_list):
    print(f"[{i+1}/{len(hr_list)}] 🔽 {hr_path}")

    # HR 이미지 불러오기
    hr = read_image(hr_path)              # Tensor [3, H, W]
    h, w = hr.shape[1:]

    # 논문 방식 다운샘플링: Gaussian blur + Bicubic resize
    blurred = gaussian_blur(hr, sigma=sigma)
    lr = resize_bicubic(blurred, h // scale, w // scale)

    # 저장
    filename = os.path.basename(hr_path)
    save_path = os.path.join(output_dir, filename)
    write_image(save_path, lr)

print(f"\n✅ FSRCNN 방식으로 다운샘플된 LR PNG 저장 완료!")
print(f"📁 저장 위치: {output_dir}")
