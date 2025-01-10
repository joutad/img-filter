import os
from sys import argv
import warp as wp
import numpy as np
from PIL import Image
from sharpen import unsharpMasking
from noiseRemove import medianFilter

wp.init()
device = "cpu"

if len(argv) != 6:
    print("\033[31mERROR: format must be:\033[0m\n\tpython3 main.py <algType> <kernSize> <param> <inFileName> <outFileName>\nexamples:\033[32m\t      -n or -s    e.g. 3   e.g. 1   sample.png   result.png\033[0m\n")
    exit(1)

algType = argv[1]
kernSize = int(argv[2])
param = float(argv[3])
inFileName = argv[4]
outFileName = argv[5]

if algType != '-n' and algType != '-s':
    print("\033[31mERROR: Invalid algType - choose between '-n' or '-s'\033[0m\n")
    exit(1)

image: Image.Image
try:
    image = Image.open(inFileName)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "L" and image.mode != "RGB":
        print("\033[31mERROR: image mode is not grayscale or RGB\033[0m\n")
        exit(1)

except FileNotFoundError:
    print("\033[31mERROR: inFileName is either not a valid image or not in the directory\033[0m\n")
    exit(1)

ext = os.path.splitext(outFileName)[1]
if not ext or len(ext) <= 1:
    print('\033[31mERROR: outFileName does not specify an image extension\033[0m\n')
    exit(1)

if kernSize % 2 == 0 or kernSize <=0 :
    print("\033[31mERROR: kernSize must be a positive odd integer\033[0m\n")
    exit(1)

if param < 0:
    print("\033[31mERROR: param must be a non-negative number\033[0m\n")
    exit(1)

if algType == '-n':
    processedImage = medianFilter(image, kernSize, device=device)
    print("\nNoise Removal: ", end="")
elif algType == '-s':
    sigma = kernSize / 6
    processedImage = unsharpMasking(image, kernSize, sigma, param, device=device)
    print("\nSharpen: ", end="")

processedImage.save(outFileName)
print("\033[93m" + inFileName, "\033[0m->", "\033[32m" + outFileName + "\033[0m\n")
