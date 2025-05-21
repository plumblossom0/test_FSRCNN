from utils.common import *
from model import FSRCNN
import argparse

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------
scale = 3
sigma = 0.2

ckpt_path_base = "C:/codelib/git/test_FSRCNN/models/FSRCNN-x3_conv/FSRCNN-x3.pt"
ckpt_path_my   = "C:/codelib/git/test_FSRCNN/models/FSRCNN-x3_human/FSRCNN-x3.pt"
data_dir   = "C:/codelib/git/test_FSRCNN/dataset/test/x3/data"
label_dir  = "C:/codelib/git/test_FSRCNN/dataset/test/x3/labels"
# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # 두 모델 불러오기
    model_base = FSRCNN(scale, device)
    model_base.load_weights(ckpt_path_base)

    model_my = FSRCNN(scale, device)
    model_my.load_weights(ckpt_path_my)

    ls_data = sorted_list(data_dir)
    ls_labels = sorted_list(label_dir)

    psnr_sum_base = 0
    psnr_sum_my = 0
    
    sum_psnr = 0
    with torch.no_grad():
        for i in range(0, len(ls_data)):
            lr_image = read_image(ls_data[i])
            lr_image = gaussian_blur(lr_image, sigma=sigma)
            hr_image = read_image(ls_labels[i])

            lr_image = rgb2ycbcr(lr_image)
            hr_image = rgb2ycbcr(hr_image)

            lr_image = norm01(lr_image)
            hr_image = norm01(hr_image)

            lr_input = torch.unsqueeze(lr_image, dim=0).to(device)
            
             # --- 예측 ---
            sr_base = model_base.predict(lr_input)[0].cpu()
            sr_my   = model_my.predict(lr_input)[0].cpu()

            # --- PSNR 계산 ---
            psnr_base = PSNR(hr_image, sr_base, max_val=1).item()
            psnr_my   = PSNR(hr_image, sr_my, max_val=1).item()

            psnr_sum_base += psnr_base
            psnr_sum_my   += psnr_my

            print(f"[{i+1}/{len(ls_data)}] PSNR(FSRCNN): {psnr_base:.3f} dB | PSNR(MyFSRCNN): {psnr_my:.3f} dB")
   
    print(f"\nAverage PSNR(FSRCNN):    {psnr_sum_base / len(ls_data):.4f} dB")
    print(f"Average PSNR(MyFSRCNN):  {psnr_sum_my   / len(ls_data):.4f} dB")

if __name__ == "__main__":
    main()

