from skimage import img_as_float 
from skimage.measure import compare_psnr #as psnr
from skimage.measure import compare_ssim #as ssim

def psnr(img1,img2):
	return compare_psnr(img_as_float(img1),img_as_float(img2))
def ssim(img1,img2):
	return compare_ssim(img_as_float(img1),img_as_float(img2),multichannel=True)
