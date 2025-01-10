from PIL import Image
import warp as wp
import numpy as np

@wp.kernel
def applyGaussianInKernel(data: wp.array3d(dtype=float), kernel: wp.array2d(dtype=float), output: wp.array3d(dtype=float), kernRadius: int, height: int, width: int, channels: int):
    i, j, k = wp.tid()

    if i < height and j < width and k < channels:
        value = float(0.0)

        # gather neighbour indices
        for navI in range(-kernRadius, kernRadius + 1):
            for navJ in range(-kernRadius, kernRadius + 1):

                neighbourI = i + navI
                neighbourJ = j + navJ

                while neighbourI < 0 or neighbourI >= height:
                    if neighbourI < 0:
                        neighbourI = -neighbourI
                    elif neighbourI >= height:
                        neighbourI = 2 * (height - 1) - neighbourI

                while neighbourJ < 0 or neighbourJ >= width:
                    if neighbourJ < 0:
                        neighbourJ = -neighbourJ
                    elif neighbourJ >= width:
                        neighbourJ = 2 * (width - 1) - neighbourJ

                value += data[neighbourI, neighbourJ, k] * kernel[navI + kernRadius, navJ + kernRadius] # convolution

        output[i, j, k] = value

def applyGaussian(imageData, mode: str, kernel, kernRadius: int, device: str) -> np.ndarray:
    if mode == "L":  # grayscale
        imageData = imageData[:, :, np.newaxis] # add axis for uniformity in warp kernel

    height, width, channels = imageData.shape
    
    data = wp.array(imageData, dtype=float, device=device)
    kern = wp.array(kernel, dtype=float, device=device)
    radius = wp.constant(kernRadius)
    output = wp.zeros_like(data)

    wp.launch(applyGaussianInKernel, dim=(height, width, channels), inputs=[data, kern, output, radius, height, width, channels], device=device)

    output = output.numpy()
    if mode == "L": # grayscale
        output = output[:, :, 0] # remove axis after kernel complete

    return output

def gaussianKernel(kernSize: int, kernRadius: int, sigma: float) -> np.ndarray:
    i, j = np.meshgrid(np.arange(-kernRadius, kernRadius + 1), np.arange(-kernRadius, kernRadius + 1))
    kernel = np.exp(-(i**2 + j**2) / (2 * sigma**2))

    return kernel / kernel.sum()

def unsharpMasking(image: Image.Image, kernSize: int, sigma: float, param: float, device: str) -> Image.Image:
    imageData = np.array(image).astype(np.float32)
    kernRadius = int((kernSize - 1) / 2)

    kernel = gaussianKernel(kernSize, kernRadius, sigma).astype(np.float32)
    blurredData = applyGaussian(imageData, image.mode, kernel, kernRadius, device=device)

    mask = imageData - blurredData
    sharpenedData = imageData + param * mask

    sharpenedData = np.clip(sharpenedData, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(sharpenedData)
