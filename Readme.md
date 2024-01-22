# Grounded-ControlNet Instructions

## Generate Dataset.
1. Setup [GLIP](https://github.com/microsoft/GLIP) for generating grounding data.
2. Setup [CLIP](https://github.com/openai/CLIP.git) for choosing best caption among available.
3. Install [ControlNet_AUX](https://github.com/patrickvonplaten/controlnet_aux) if you want openpose style skeleton support, else use the COCO skeletons.
4. Download [COCO2014](https://cocodataset.org/#download).
5. Clone this repo.
5. Use `make_dataset_grounded.py` to generate dataset. This generates `coco14.json` for a HuggingFace dataset.
6. Use `coco14.py` from the same to repo and combine with generated `coco14.json` to generated HuggingFace Dataset.

## Grounded ControlNet.

1. Source is available [here](https://github.com/ravich3373/diffusers_fork), do an editable install `pip install -e .`.
2. Pipeline is avialable as `StableDiffusionControlNetGroundedPipeline`, `from diffusers import StableDiffusionControlNetGroundedPipeline`.
3. Grounded ControlNet model is availabel as `ControlNetGroundedModel`, `from diffusers import ControlNetGroundedModel`.
4. Training script is available as `examples/controlNet/train_controlnet_grounded.py`, usage is similar to `train_controlnet.py` see [here](https://huggingface.co/blog/train-your-controlnet).