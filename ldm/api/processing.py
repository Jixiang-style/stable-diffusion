import os
import logging
import random
import string
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.api.config import Config
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def select_seed_randomly() -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    return random.randint(min_seed_value, max_seed_value)


class Processing:
    logging.basicConfig(handlers=[RotatingFileHandler(filename=Config.LOG_FILE, encoding='utf-8', mode='a',
                                                      maxBytes=52428800, backupCount=10)],
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("zeep").setLevel(logging.WARNING)

    config = OmegaConf.load(f"{Config.CONFIG_FILE}")
    device = torch.device("cuda") if Config.DEVICE == "cuda" else torch.device("cpu")
    start_time = time.time()
    model = load_model_from_config(config, f"{Config.CKPT_PATH}", device)

    @classmethod
    def txt2img(cls, opt):
        opt.precision = 'full'
        start_time = time.time()
        img_path = cls.main(opt)
        logging.info("[txt2img] Execution time: %s seconds" % (time.time() - start_time))
        return img_path

    @classmethod
    def main(cls, opt):
        seed_everything(opt.seed)

        if opt.plms:
            sampler = PLMSSampler(cls.model, device=cls.device)
        elif opt.dpm:
            sampler = DPMSolverSampler(cls.model, device=cls.device)
        else:
            sampler = DDIMSampler(cls.model, device=cls.device)

        date_time = datetime.now().strftime("%Y%m%d%H%M%S")

        outpath = Config.TXT2IMG_OUT_DIR + '/' + date_time[0:8]
        os.makedirs(outpath, exist_ok=True)

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = [p for p in data for i in range(opt.repeat)]
                data = list(chunk(data, batch_size))

        # sample_path = os.path.join(outpath, "samples")
        # os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        # base_count = len(os.listdir(sample_path))
        # grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=cls.device)

        if opt.torchscript or opt.ipex:
            transformer = cls.model.cond_stage_model.model
            unet = cls.model.model.diffusion_model
            decoder = cls.model.first_stage_model.decoder
            additional_context = torch.cpu.amp.autocast() if opt.bf16 else nullcontext()
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

            if opt.bf16 and not opt.torchscript and not opt.ipex:
                raise ValueError('Bfloat16 is supported only for torchscript+ipex')
            if opt.bf16 and unet.dtype != torch.bfloat16:
                raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                                 "you'd like to use bfloat16 with CPU.")
            if unet.dtype == torch.float16 and cls.device == torch.device("cpu"):
                raise ValueError(
                    "Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

            if opt.ipex:
                import intel_extension_for_pytorch as ipex
                bf16_dtype = torch.bfloat16 if opt.bf16 else None
                transformer = transformer.to(memory_format=torch.channels_last)
                transformer = ipex.optimize(transformer, level="O1", inplace=True)

                unet = unet.to(memory_format=torch.channels_last)
                unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

                decoder = decoder.to(memory_format=torch.channels_last)
                decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            if opt.torchscript:
                with torch.no_grad(), additional_context:
                    # get UNET scripted
                    if unet.use_checkpoint:
                        raise ValueError("Gradient checkpoint won't work with tracing. " +
                                         "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                    img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                    t_in = torch.ones(2, dtype=torch.int64)
                    context = torch.ones(2, 77, 1024, dtype=torch.float32)
                    scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                    scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                    print(type(scripted_unet))
                    cls.model.model.scripted_diffusion_model = scripted_unet

                    # get Decoder for first stage model scripted
                    samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                    scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                    scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                    print(type(scripted_decoder))
                    cls.model.first_stage_model.decoder = scripted_decoder

            prompts = data[0]
            print("Running a forward pass to initialize optimizations")
            uc = None
            if opt.scale != 1.0:
                uc = cls.model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            with torch.no_grad(), additional_context:
                for _ in range(3):
                    c = cls.model.get_learned_conditioning(prompts)
                samples_ddim, _ = sampler.sample(S=5,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=start_code)
                print("Running a forward pass for decoder")
                for _ in range(3):
                    x_samples_ddim = cls.model.decode_first_stage(samples_ddim)

        precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext
        result = None
        with torch.no_grad(), \
                precision_scope(cls.device), \
                cls.model.ema_scope():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = cls.model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = cls.model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)

                    x_samples = cls.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        # img = Image.fromarray(x_sample.astype(np.uint8))
                        # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        # base_count += 1
                        sample_count += 1

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
            grid_path = os.path.normpath(os.path.join(outpath, f'grid-{date_time}-{random_str}.png'))
            grid.save(grid_path)
            # grid_count += 1
            result = os.path.abspath(grid_path)

        print(f"Your samples are ready and waiting for you here: {result}")
        return result
