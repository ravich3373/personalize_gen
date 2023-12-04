import pandas as pd
from huggingface_hub import hf_hub_url
import datasets
import os
import json
from pathlib import Path


COCO_IMGS_PTH = ""
SEL_JSON_PTH = ""
KP_CTRL_PTH = ""


_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

METADATA_URL = hf_hub_url(
    "fusing/fill50k",
    filename="train.jsonl",
    repo_type="dataset",
)

IMAGES_URL = hf_hub_url(
    "fusing/fill50k",
    filename="images.zip",
    repo_type="dataset",
)

CONDITIONING_IMAGES_URL = hf_hub_url(
    "fusing/fill50k",
    filename="conditioning_images.zip",
    repo_type="dataset",
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class COCO14(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = SEL_JSON_PTH
        images_dir = COCO_IMGS_PTH
        conditioning_images_dir = KP_CTRL_PTH

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        with open(metadata_path) as fp:
            metadata = json.load(fp)

        for imgid, sample in metadata.items():
            text = sample["caption"]

            fn = Path(sample["file_name"])
            image_path = os.path.join(images_dir, fn)
            image = open(image_path, "rb").read()

            conditioning_image_path = f"{fn.stem}_kp{fn.suffix}"
            conditioning_image_path = os.path.join(
                conditioning_images_dir, conditioning_image_path
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield image_path, {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }
