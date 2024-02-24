import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from pathlib import Path


from .humanparsing.aigc_run_parsing import Parsing
from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location
from .openpose.run_openpose import OpenPose


_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}


class OOTDiffusionModel:
    def __init__(self, hg_root: str, cache_dir: str = None):
        self.model = OOTDiffusion(
            hg_root=hg_root,
            cache_dir=cache_dir,
        )

    def generate(self, pipe, cloth_path: str | bytes | Path, model_path: str | bytes | Path, seed=0, steps=1, cfg=1.0):
        category = "upperbody"
        # if model_image.shape != (1, 1024, 768, 3) or (
        #     cloth_image.shape != (1, 1024, 768, 3)
        # ):
        #     raise ValueError(
        #         f"Input image must be size (1, 1024, 768, 3). "
        #         f"Got model_image {model_image.shape} cloth_image {cloth_image.shape}"
        #     )

        # (1,H,W,3) -> (3,H,W)
        model_image = Image.open(model_path).resize((768, 1024))
        cloth_image = Image.open(cloth_path).resize((768, 1024))

        model_parse, _ = Parsing(pipe.device)(model_image.resize((384, 512)))
        keypoints = OpenPose()(model_image.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            pipe.model_type,
            _category_get_mask_input[category],
            model_parse,
            keypoints,
            width=384,
            height=512,
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_image, mask)
        images = pipe(
            category=category,
            image_garm=cloth_image,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_image,
            num_samples=1,
            num_steps=steps,
            image_scale=cfg,
            seed=seed,
        )

        output_image = to_tensor(images[0])
        output_image = output_image.permute((1, 2, 0))
        masked_vton_img = masked_vton_img.convert("RGB")
        masked_vton_img = to_tensor(masked_vton_img)
        masked_vton_img = masked_vton_img.permute((1, 2, 0))

        return ([output_image], [masked_vton_img])

    def load(self):
        return self.model

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return str(self.model)
