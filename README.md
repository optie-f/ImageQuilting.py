# ImageQuilting. py

This is unofficial Python implementation of *[Image Quilting for texture synthesis and transfer [Efros & Freeman, 2001]](https://people.eecs.berkeley.edu/~efros/research/quilting.html)* for my practice.

It loads images in `tex/` as inputs, and outputs will be rendered in `result/`.

Currently only texture synthesis is implemented. (style transfer is yet)

# Results

Sample textures are from textures.com.

## Input 
(256 * 256) 

![input_1](example/0001.jpg)
## Output 
(1024 * 1024) 

![output_1](example/0001_size1024_patch64.jpg)
Minimum error lines:

![output_1_line](example/0001_size1024_patch64_line.jpg)

---
## Input 
(256 * 256)

![input_1](example/0002.jpg)
## Output 

(1024 * 1024)
![output_1](example/0002_size1024_patch64.jpg)
Minimum error lines:

![output_1_line](example/0002_size1024_patch64_line.jpg)
