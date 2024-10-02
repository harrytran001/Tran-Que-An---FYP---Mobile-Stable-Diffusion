# Stable Diffusion-NCNN

Stable-Diffusion implemented by [ncnn](https://github.com/Tencent/ncnn) framework based on C++, supported txt2img and img2img!

Zhihu: https://zhuanlan.zhihu.com/p/582552276

Video: https://www.bilibili.com/video/BV15g411x7Hc

***txt2img Performance (time pre-it and ram)***
| per-it | i7-12700 (512x512)  | i7-12700 (256x256) | Snapdragon865 (256x256) |
| ------ | ------------------- | ------------------ | ----------------------- |
| slow   | 4.85s/5.24G(7.07G)  | 1.05s/3.58G(4.02G) | 1.6s/2.2G(2.6G)         |
| fast   | 2.85s/9.47G(11.29G) | 0.65s/5.76G(6.20G) |                         |

## News

2023-03-11: happy to add img2img android and release new apk

2023-03-10: happy to add img2img x86

2023-01-19: speed up & less ram in x86, dynamic shape in x86

2023-01-12: update to the latest ncnn code and use optimize model, update android, add memory monitor

2023-01-05: add 256x256 model to x86 project

2023-01-04: merge and finish the mha op in x86, enable fast gelu

## Demo

![image](./resources/image.png)

<p align="center">
  <img src="./resources/android.jpg" width="320x">
</p>

## Out of box

All models and exe file you can download from [百度网盘](https://pan.baidu.com/s/1Q_p0N3v7Y526Ht3JbxJ1XQ?pwd=6666) or [Google Drive](https://drive.google.com/drive/folders/1myB4uIQ2K5okl51XDbmYhetLF9rUyLZS?usp=sharing) or Release

If you only need ncnn model, you can search it from [硬件模型库-设备专用模型](https://platform.openmmlab.com/deploee), it would be more faster and free.

### x86 Windows
1. enter folder [exe](./x86/exe)
2. download 4 bin file: ```AutoencoderKL-fp16.bin, FrozenCLIPEmbedder-fp16.bin, UNetModel-MHA-fp16.bin, AutoencoderKL-encoder-512-512-fp16.bin``` and put them to ```assets``` folder
3. set up your config in ```magic.txt```, each line are:
    1. height (must be a multiple of 128, minimum is 256)
    2. width (must be a multiple of 128, minimum is 256)
    3. speed mode (0 is slow but low ram, 1 is fast but high ram)
    4. step number (15 is not bad)
    5. seed number (set 0 to be random)
    6. init image (if the file is exist, run img2img, if not, run txt2img)
    7. positive prompt (describe what you want)
    8. negative prompt (describe what you don't want)
4. run ```stable-diffusion.exe```

### android apk
1. download an install the apk from the link
2. in the top, the first one is step and the second one is seed
3. int the bottom, the top one the positive prompt and the bottom one negative prompt (set empty to enable the default prompt)
4. note: the apk needs 7G ram, and run very slow and power consumption

## Implementation Details

Note: Please comply with the requirements of the SD model and do not use it for illegal purposes

1. Three main steps of Stable-Diffusion：
    1. CLIP: text-embedding
    2. (only img2img) encode the init image to init latent
    3. iterative sampling with sampler
    4. decode the sampler results to obtain output images
2. Model details：
    1. Weights：Naifu (u know where to find)
    2. Sampler：Euler ancestral (k-diffusion version)
    3. Resolution：dynamic shape, but must be a multiple of 128, minimum is 256
    4. Denoiser：CFGDenoiser, CompVisDenoiser
    4. Prompt：positive & negative, both supported :)

## Code Details

### Complie for x86 Windows
1. download 4 bin file: ```AutoencoderKL-fp16.bin, FrozenCLIPEmbedder-fp16.bin, UNetModel-MHA-fp16.bin, AutoencoderKL-encoder-512-512-fp16.bin``` and put them to ```assets``` folder
2. open the vs2019 project and compile the release&x64

### Complie for x86 Linux / MacOS

1. [build and Install NCNN](https://github.com/Tencent/ncnn/wiki/how-to-build#pass-for-linux)
2. build the demo with CMake

```sh
cd x86/linux
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

3. download 3 bin file: ```AutoencoderKL-fp16.bin, FrozenCLIPEmbedder-fp16.bin, UNetModel-MHA-fp16.bin``` and put them to `build/assets` folder
4. run the demo

```sh
./stable-diffusion-ncnn
```

### Compile for android
1. download three bin file: ```AutoencoderKL-fp16.bin, FrozenCLIPEmbedder-fp16.bin, UNetModel-MHA-fp16.bin``` and put them to ```assets``` folder
2. open android studio and run the project

### ONNX Model

I've uploaded the three onnx models used by Stable-Diffusion, so that you can do some interesting work.

You can find them from the link above.

#### Statements

1. Please abide by the agreement of the stable diffusion model consciously, and DO NOT use it for illegal purposes!
2. If you use these onnx models to make open source projects, please inform me and I'll follow and look forward for your next great work :)

#### Instructions

1. FrozenCLIPEmbedder

```C++
ncnn (input & output): token, multiplier, cond, conds
onnx (input & output): onnx::Reshape_0, 2271

z = onnx(onnx::Reshape_0=token)
origin_mean = z.mean()
z *= multiplier
new_mean = z.mean()
z *= origin_mean / new_mean
conds = torch.concat([cond,z], dim=-2)
```

2. UNetModel

```C++
ncnn (input & output): in0, in1, in2, c_in, c_out, outout
onnx (input & output): x, t, cc, out

outout = in0 + onnx(x=in0 * c_in, t=in1, cc=in2) * c_out
```

## References

1. [ncnn](https://github.com/Tencent/ncnn)
2. [opencv-mobile](https://github.com/nihui/opencv-mobile)
3. [stable-diffusion](https://github.com/CompVis/stable-diffusion)
4. [k-diffusion](https://github.com/crowsonkb/k-diffusion)
5. [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
6. [diffusers](https://github.com/huggingface/diffusers)
7. [diffusers-ncnn](https://github.com/EdVince/diffusers-ncnn)
