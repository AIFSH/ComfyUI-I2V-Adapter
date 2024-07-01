import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

import torch
import time
import shutil
from PIL import Image
import cuda_malloc
import folder_paths
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download, hf_hub_download

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from i2v_adapter.models.ip_adapter import Resampler
from i2v_adapter.models.unet import UNet3DConditionModel
from i2v_adapter.pipelines.pipeline_i2v_adapter import I2VIPAdapterPipeline
from i2v_adapter.utils.util import save_videos_grid, load_weights, imread_resize, color_match_frames,resize_image

output_dir = folder_paths.get_output_directory()
ckpts_dir = os.path.join(now_dir,"pretrained_models")
pretrained_model_path = os.path.join(ckpts_dir, "stable-diffusion-v1-5")
I2V_Adapter_dir = os.path.join(ckpts_dir,"I2V-Adapter")
pretrained_image_encoder_path = os.path.join(ckpts_dir,"IP-Adapter","models","image_encoder")
pretrained_ipadapter_path = os.path.join(ckpts_dir,"IP-Adapter","models","ip-adapter-plus_sd15.bin")
i2v_module_path = os.path.join(I2V_Adapter_dir,"i2v_module.pth")
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

class PromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "get_text"

    CATEGORY = "AIFSH_I2V-Adapter"

    def get_text(self,text):
        return (text, )

class LoraPathLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "lora_name": (folder_paths.get_filename_list("loras"), ),
            "lora_weight":("FLOAT",{
                "min":0.,
                "max":1.0,
                "step":0.01,
                "default":0.8,
                "display":"silder"
            })
            }}
    RETURN_TYPES = ("LORAMODULE",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "AIFSH_I2V-Adapter"

    def load_checkpoint(self, lora_name,lora_weight):
        ckpt_path = folder_paths.get_full_path("loras", lora_name)
        lora_module = {
            "lora_model_path":ckpt_path,
            "lora_alpha":lora_weight,
        }
        return (lora_module,)
    

class MotionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "motion_type":(["PanLeft","PanRight","RollingAnticlockwise",
                               "RollingClockwise","TiltDown","TiltUp","ZoomIn","ZoomOut"],),
                "motion_wight":("FLOAT",{
                    "min":0.,
                    "max":1.0,
                    "step":0.01,
                    "default":0.8,
                    "display":"silder"
                })
            }
        }
    
    RETURN_TYPES = ("MOTIONLORA",)
    FUNCTION = "get_motion_lora"

    CATEGORY = "AIFSH_I2V-Adapter"

    def get_motion_lora(self,motion_type,motion_wight):
        filename = f"v2_lora_{motion_type}.ckpt"
        motion_local_dir = os.path.join(ckpts_dir,"motion_lora")
        motion_path = os.path.join(motion_local_dir, filename)
        if not os.path.isfile(motion_path):
            hf_hub_download(repo_id="guoyww/animatediff",filename=filename,local_dir=motion_local_dir)
        motion_module = [{
            "path":motion_path,
            "alpha":motion_wight
        }]
        return (motion_module,)
    
class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_I2V-Adapter"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

class I2V_AdapterNode:

    def __init__(self):
        self.pipe = None
        # jcplus/stable-diffusion-v1-5
        snapshot_download(repo_id="runwayml/stable-diffusion-v1-5",local_dir=pretrained_model_path,
                          ignore_patterns=["*-pruned*","*.bin","*fp16*","*ckpt","*non_ema*"])
        # space-xun/i2v_adapter
        if not os.path.isfile(os.path.join(I2V_Adapter_dir,"i2v_module.pth")):
            snapshot_download(repo_id="space-xun/i2v_adapter",local_dir=I2V_Adapter_dir)
            # move animatediff_v15_v1_ipplus.pth
            shutil.move(os.path.join(I2V_Adapter_dir,"animatediff_v15_v1_ipplus.pth"),os.path.join(pretrained_model_path,"unet","animatediff_v15_v1_ipplus.pth"))
        # h94/IP-Adapter
        hf_hub_download(repo_id="h94/IP-Adapter",filename="ip-adapter-plus_sd15.bin",subfolder="models",local_dir=os.path.join(ckpts_dir,"IP-Adapter"))
        # hf_hub_download(repo_id="h94/IP-Adapter",filename="config.json",subfolder="models/image_encoder",local_dir=os.path.join(ckpts_dir,"IP-Adapter"))
        # hf_hub_download(repo_id="h94/IP-Adapter",filename="pytorch_model.bin",subfolder="models/image_encoder",local_dir=os.path.join(ckpts_dir,"IP-Adapter"))
        
        snapshot_download(repo_id="h94/IP-Adapter",local_dir=os.path.join(ckpts_dir,"IP-Adapter"),allow_patterns=["models/*","pytorch_model.bin"],ignore_patterns=["*.safetensors","*adapter*"])
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "prompt":("TEXT",),
                "neg_prompt":("TEXT",),
                "resolution":([256,512],{
                    "default": 512
                }),
                "video_length":("INT",{
                    "default": 16,
                }),
                "fps":("INT",{
                    "default": 8,
                }),
                "num_inference_steps":("INT",{
                    "default": 25,
                }),
                "cfg":("FLOAT",{
                    "default": 7.5,
                }),
                "seed":("INT",{
                    "default":42
                })
            },
            "optional":{
                "lora_module": ("LORAMODULE",),
                "motion_lora":("MOTIONLORA",),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_I2V-Adapter"

    def gen_video(self,image,prompt,neg_prompt,resolution,video_length,fps,
                  num_inference_steps,cfg,seed,lora_module=None,motion_lora=None):
        image_np = image.numpy()[0] * 255
        image_np = image_np.astype(np.uint8)
        # load models
        inference_config = OmegaConf.load(os.path.join(now_dir,"infer.yaml"))
        global_seed = seed

        if self.pipe is None:
            unet = UNet3DConditionModel.from_pretrained_ip(pretrained_model_path, subfolder="unet",
                                                            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(torch.float16).to(device)
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=torch.float16)
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
            clip_image_processor = CLIPImageProcessor()
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_image_encoder_path, torch_dtype=torch.float16).to(device)
            image_proj_model = Resampler(
                    dim=768,
                    depth=4,
                    dim_head=64,
                    heads=12,
                    num_queries=16,
                    embedding_dim=image_encoder.config.hidden_size,
                    output_dim=768,
                    ff_mult=4
                )
            image_proj_model.load_state_dict(torch.load(pretrained_ipadapter_path, "cpu")["image_proj"], strict=True)
            image_proj_model.to(torch.float16).to(device)
            print("Load pretrained clip image encoder and ipadapter model successfully")
            
            
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
                
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)
            unet.requires_grad_(False)
            image_encoder.requires_grad_(False)
            image_proj_model.requires_grad_(False)

            # builds pipeline
            noise_scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
            self.pipe = I2VIPAdapterPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
                                        scheduler=noise_scheduler)



        if lora_module is None:
            lora_module = {
                "lora_model_path":"",
                "lora_alpha":0.8,
            }
        pipe = load_weights(self.pipe, 
                            i2v_module_path=i2v_module_path,
                            motion_module_lora_configs=motion_lora if motion_lora else [],
                            **lora_module)
        pipe.to(torch.float16).to(device)
        pipe.enable_vae_slicing()


        ## gen video
        print('Prompt: ', prompt)
        print('Negative Prompt: ', neg_prompt)

        image_pil = Image.fromarray(image_np)
        image = clip_image_processor(images=image_pil, return_tensors="pt").pixel_values.to(device).to(torch.float16)

        with torch.no_grad():
            clip_image_embeds = image_encoder(image, output_hidden_states=True).hidden_states[-2]
            clip_image_embeds = image_proj_model(clip_image_embeds)
            un_cond_image_embeds = image_encoder(torch.zeros_like(image).to(image.device).to(torch.float16), output_hidden_states=True).hidden_states[-2]
            un_cond_image_embeds = image_proj_model(un_cond_image_embeds)
    
        print("Using seed {} for generation".format(global_seed))
        generator = torch.Generator(device="cuda").manual_seed(global_seed)
        # Get first frame latents as usual
        # image = imread_resize(img_path, args.height, args.width)
        image = resize_image(image_np,resolution)
        first_frame_latents = torch.Tensor(image.copy()).to(device).type(torch.float16).permute(2, 0, 1).repeat(1, 1, 1, 1)
        first_frame_latents = first_frame_latents / 127.5 - 1.0
        first_frame_latents = vae.encode(first_frame_latents).latent_dist.sample(generator) * 0.18215
        first_frame_latents = first_frame_latents.repeat(1, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
        
        # video generation
        video = pipe(prompt=prompt, generator=generator, latents=first_frame_latents, 
                    video_length=video_length, height=image.shape[0], width=image.shape[1], 
                    num_inference_steps=num_inference_steps, guidance_scale=cfg, 
                    noise_mode="iid", negative_prompt=neg_prompt, 
                    repeat_latents=True, gaussian_blur=True,
                    cond_image_embeds=clip_image_embeds,
                    un_cond_image_embeds=un_cond_image_embeds).videos

        # histogram matching post processing
        for f in range(1, video.shape[2]):
            former_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            frame = video[0, :, f, :, :].permute(1, 2, 0).cpu().numpy()
            result = color_match_frames(former_frame, frame)
            result = torch.Tensor(result).type_as(video).to(video.device)
            video[0, :, f, :, :] = result.permute(2, 0, 1)
        print(video.shape)
        # save_path = args.output
        save_path = os.path.join(output_dir, f"i2v_adapter_{time.time_ns()}.mp4")
        save_videos_grid(video, save_path,fps=fps)
        return (save_path,)

