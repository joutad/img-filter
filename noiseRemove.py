import warp as wp
import numpy as np
from PIL import Image

@wp.kernel
def applyMedian(data: wp.array3d(dtype=float), output: wp.array3d(dtype=float), neighbourhood: wp.array(dtype=float), kernRadius: int, height: int, width: int, channels: int):
    i, j, k = wp.tid()

    if i < height and j < width and k < channels:
        count = int(0)

        # establish neighbourhood list
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

                neighbourhood[count] = data[neighbourI, neighbourJ, k]
                count += 1

        # sorting for median
        for index in range(1, count):
            key = neighbourhood[index]
            l = index - 1
            while l >= 0 and neighbourhood[l] > key:
                neighbourhood[l + 1] = neighbourhood[l]
                l -= 1
            neighbourhood[l + 1] = key

        output[i, j, k] = neighbourhood[count // 2]

def medianFilter(image: Image.Image, kernSize: int, device: str) -> Image.Image:
    imageData = np.array(image).astype(np.float32)
    kernRadius = int((kernSize - 1) / 2)

    if image.mode == "L": # grayscale
        imageData = imageData[:, :, np.newaxis] # add axis for uniformity in warp kernel

    height, width, channels = imageData.shape

    data = wp.array(imageData, dtype=float, device=device)
    output = wp.zeros_like(data)
    neighbourhood = wp.zeros(kernSize * kernSize, dtype=float, device=device)
    radius = wp.constant(kernRadius)

    wp.launch(applyMedian, dim=(height, width, channels), inputs=[data, output, neighbourhood, radius, height, width, channels], device=device)

    output = output.numpy().astype(np.uint8)

    if image.mode == "L": # grayscale
        output = output[:, :, 0] # remove axis after kernel complete

    return Image.fromarray(output)
