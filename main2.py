import torch
import argparse
from PIL import Image
import numpy as np
import os
from omegaconf import OmegaConf

import ImageBind.data as data
from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType

from ldm.models.diffusion.ddpm import ImageEmbeddingConditionedLatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler


class Binder:
    """ Wrapper for ImageBind model
    """
    def __init__(self, pth_path, device='cuda'):
        self.model = imagebind_model.imagebind_huge_pth(pretrained=True, pth_path=pth_path)
        self.device = device
        self.model.eval()
        self.model.to(device)

        self.data_process_dict = {ModalityType.TEXT: data.load_and_transform_text,
                                  ModalityType.VISION: data.load_and_transform_vision_data,
                                  ModalityType.AUDIO: data.load_and_transform_audio_data}

    def run(self, ctype, cpaths):
        """ ctype: str
            cpaths: list[str]
        """
        inputs = {ctype: self.data_process_dict[ctype](cpaths, self.device)}
        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings[ctype]


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bind_path', type=str, default="ImageBind/.checkpoints/imagebind_huge.pth", help="path to imagebind model")
    argparser.add_argument('--config', type=str, default='stablediffusion/configs/stable-diffusion/v2-1-stable-unclip-h-inference.yaml', help="path to diffusion config")
    argparser.add_argument('--ckpt', type=str, default='stablediffusion/checkpoints/sd21-unclip-h.ckpt', help="path to diffusion model")
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--H', type=int, default=768, help="height of output image")
    argparser.add_argument('--W', type=int, default=768, help="width of output image")
    argparser.add_argument('--f', type=int, default=8, help="downsample factor of latent image")
    argparser.add_argument('--C', type=int, default=4, help="number of channels of latent image")
    argparser.add_argument('--steps', type=int, default=50, help="number of steps of DDIM")
    argparser.add_argument('--n_samples', type=int, default=2)
    argparser.add_argument('--scale', type=int, default=10, help="scale of guidance")
    argparser.add_argument('--ctype', type=str, default='audio', help="modality type")
    argparser.add_argument('--cpath', type=str, default='cond_examples/dog_audio.wav', help="modality paths")

    argparser.add_argument('--mode', type=str, default='generation', choices=['generation', 'editing'])
    argparser.add_argument('--init_img', type=str, default='sketch_bird.jpg', help="image to edit")
    argparser.add_argument('--strength', type=float, default=0.7, help="editing strength")

    argparser.add_argument('--prompt', type=str, default='', help="text prompt")
    argparser.add_argument('--negative', type=str, default='', help="negative text prompt")
    argparser.add_argument('--output_dir', type=str, default='./output', help="negative text prompt")

    # --------------------------------------
    # Parse arguments
    # --------------------------------------
    opt = argparser.parse_args()
    config = OmegaConf.load(f"{opt.config}")

    latent_shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    batch_size = opt.n_samples
    mode = opt.mode
    init_img = opt.init_img
    strength = opt.strength
    t_enc = int(strength * opt.steps)

    # --------------------------------------
    # Build models
    # --------------------------------------
    binder = Binder(pth_path=opt.bind_path, device=opt.device)
    pl_sd = torch.load(opt.ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = ImageEmbeddingConditionedLatentDiffusion(**config.model['params'])
    model.load_state_dict(sd, strict=False)
    model.to(opt.device)
    model.eval()

    sampler = DDIMSampler(model, device=opt.device)
    if mode == "editing":
        sampler.make_schedule(ddim_num_steps=opt.steps, ddim_eta=0, verbose=False)

    # --------------------------------------
    # Run generation
    # --------------------------------------
    with torch.no_grad(), torch.autocast('cuda'):
        prompts = [opt.prompt] * batch_size 
        negatives = [opt.negative] * batch_size
        c_adm = binder.run(opt.ctype, [opt.cpath])
        c_adm, noise_level_emb = model.noise_augmentor(c_adm, noise_level=torch.zeros(batch_size).long().to(c_adm.device))
        c_adm = torch.cat((c_adm, noise_level_emb), 1)

        uc = model.get_learned_conditioning(negatives)
        uc = {"c_crossattn": [uc], "c_adm": torch.zeros_like(c_adm)}
        c = model.get_learned_conditioning(prompts)
        c = {"c_crossattn": [c], "c_adm": c_adm}

        if mode == "editing":
            assert init_img is not None
            init_image = load_img(init_img).to(opt.device)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
            init_latent = torch.cat([init_latent]*batch_size, dim=0)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(opt.device))

            # decode it
            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                     unconditional_conditioning=uc,)
        else:
            samples, _ = sampler.sample(S=opt.steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=latent_shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,
                                        eta=0,
                                        x_T=None)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        x_samples = x_samples.permute(0,2,3,1).cpu().float().numpy()

    
    os.makedirs(opt.output_dir, exist_ok=True)
    for i in range(batch_size):
        output = Image.fromarray((x_samples[i] * 255).astype('uint8'))
        output.save(os.path.join(opt.output_dir, f'output_{i}.png'))
