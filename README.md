# ComfyUI-I2V-Adapter
a comfyui custom node for [I2V-Adapter](https://github.com/KwaiVGI/I2V-Adapter), the
[workflow](./doc/i2v_adapter_base_workflow.json) can be find in `doc`

## Example
- input
<div>
  <figure>
  <img alt='Wechat' src="./doc/test.png?raw=true" width="512px"/>
  <figure>
</div>

- prmopt
```
an anime girl with long brown hair hugging a white cat
```

- output

https://github.com/AIFSH/ComfyUI-I2V-Adapter/assets/149982694/203d8812-1d14-422b-89a9-3b24e53455dc

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
## insatll xformers match your torch,for torch==2.1.0+cu121
pip install xformers==0.0.22.post7
pip install accelerate 
# in ComfyUI/custom_nodes
git clone https://github.com/AIFSH/ComfyUI-I2V-Adapter.git
cd ComfyUI-I2V-Adapter
pip install -r requirements.txt
```
weights will be downloaded from huggingface

## Tutorial
- [Demo]()
- QQ群：852228202

## Thanks
[I2V-Adapter](https://github.com/KwaiVGI/I2V-Adapter)
