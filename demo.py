from utils.common import *
from model import FSRCNN 
import argparse

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------
image_path = "C:/codelib/git/test_FSRCNN/dataset/test1.png"
ckpt_path = "C:/codelib/git/test_FSRCNN/checkpoint/x3/FSRCNN-x3.pt"
scale = 3

if scale != 3:
    raise ValueError("Only scale=3 is supported in this version.")

sigma = 0.2


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------
os.makedirs("C:/codelib/git/test_FSRCNN/results", exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)

    write_image("C:/codelib/git/test_FSRCNN/results/bicubic.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    lr_image = rgb2ycbcr(lr_image)
    lr_image = norm01(lr_image)
    lr_image = torch.unsqueeze(lr_image, dim=0)

    model = FSRCNN(scale, device)
    model.load_weights(ckpt_path)
    with torch.no_grad():
        lr_image = lr_image.to(device)
        sr_image = model.predict(lr_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    sr_image = ycbcr2rgb(sr_image)

    write_image("C:/codelib/git/test_FSRCNN/results/sr.png", sr_image)

if __name__ == "__main__":
    main()
