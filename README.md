# IMG FILTER
This program performs two different image processing tasks:
- Noise removal by applying median filtering
- Sharpening images via the unsharp masking method

This program makes use of Nvidia Warp API, which allows the use of Nvidia GPUs to speed up processing of images.

See the **GPU acceleration** section to see how to improve performance if you have access to Nvidia GPUs. 

## Dependencies
Ensure you have the following python libraries installed:
- `numpy`
- `pillow` (Pillow)
- `warp`

You can install them with:

`pip install numpy pillow warp`

## Components
`main.py`:
- Main program that will call either the image sharpening or noise removal algorithm

`sharpen.py`:
- Contains the unsharp masking algorithm for sharpening images

`noiseRemove.py`:
- Contains the noise removal algorithm that makes use of median filtering

## How to run the program
The program is executed in the command line in the following format:

`python3 main.py <algType> <kernSize> <param> <inFileName> <outFileName>`

### Arguments
`<algType>`: The type of algorithm to apply:
- `-n` - Apply median filtering for noise removal.
- `-s` - Apply unsharp masking for sharpening.

`<kernSize>`:
- The size of the kernel (must be a positive odd integer, e.g., 3, 5, 7).

`<param>`: A parameter specific to the algorithm:
- For median filtering (`-n`):
  - this is a dummy value (just enter '1').
- For unsharp masking (`-s`),
  - this is the scaling factor for sharpening.

`<inFileName>`:
- Path to the input image file.

`<outFileName>`:
- Path to save the output image file (must include an image extension, e.g., .png, .jpg).

### Examples
#### Sharpening
`python3 main.py -s 3 1 image.png result.png`

#### Noise removal
`python3 main.py -n 3 1 image.png result.png`

## Troubleshooting
One issue some may run into is if the image is using unsupported channels. Make sure the image is only using the following channels:
- RGB, 
- RGBA,
- Greyscale
