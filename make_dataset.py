import json
from collections import defaultdict
from xtcocotools.coco import COCO
from tqdm.auto import tqdm
import numpy as np
import os
import clip
from PIL import Image
import torch
from pathlib import Path
import cv2

# TODO
# 1. Right now the json still has both train and val splits, unify for more train data.

SELECT_DATA = False
GEN_KP_CTRL = True
GEN_JF_JSON = True


OUT_DIR = "/scratch/rc5124/llvm/datasets/coco2014/sel_data"
coco_dir = "/scratch/rc5124/llvm/datasets/coco2014/"
KP_DIR = f"{OUT_DIR}/kp_imgs"
sel_kp_frac = 0.3

splits = ["train", "val"]
captions_fls = ["annotations/captions_train2014.json", "annotations/captions_val2014.json"]
keypoints_fls = ["annotations/person_keypoints_train2014.json", "annotations/person_keypoints_val2014.json"]



colors = [
    255,     0,    85, 
    255,     0,     0, 
    255,    85,     0, 
    255,   170,     0, 
    255,   255,     0, 
    170,   255,     0, 
     85,   255,     0, 
      0,   255,     0, 
    255,     0,     0, 
      0,   255,    85, 
      0,   255,   170, 
      0,   255,   255, 
      0,   170,   255, 
      0,    85,   255, 
      0,     0,   255, 
    255,     0,   170, 
    170,     0,   255, 
    255,     0,   255, 
     85,     0,   255]
colors = np.array(colors).reshape(-1,3)


class FastVisualizer:
    """MMPose Fast Visualizer.

    A simple yet fast visualizer for video/webcam inference.

    Args:
        metainfo (dict): pose meta information
        radius (int, optional)): Keypoint radius for visualization.
            Defaults to 6.
        line_width (int, optional): Link width for visualization.
            Defaults to 3.
        kpt_thr (float, optional): Threshold for keypoints' confidence score,
            keypoints with score below this value will not be drawn.
            Defaults to 0.3.
    """

    def __init__(self, metainfo, radius=6, line_width=3, kpt_thr=0.3):
        self.radius = radius
        self.line_width = line_width
        self.kpt_thr = kpt_thr

        self.keypoint_id2name = metainfo['keypoint_id2name']
        self.keypoint_name2id = metainfo['keypoint_name2id']
        self.keypoint_colors = metainfo['keypoint_colors']
        self.skeleton_links = metainfo['skeleton_links']
        self.skeleton_link_colors = metainfo['skeleton_link_colors']

    def draw_pose(self, img, keypoints, status):
        """Draw pose estimations on the given image.

        This method draws keypoints and skeleton links on the input image
        using the provided instances.

        Args:
            img (numpy.ndarray): The input image on which to
                draw the pose estimations.
            instances (object): An object containing detected instances'
                information, including keypoints and keypoint_scores.

        Returns:
            None: The input image will be modified in place.
        """

        keypoints = keypoints

        for kpts, use in zip(keypoints, status):
            for sk_id, sk in enumerate(self.skeleton_links):

                pos1 = (int(kpts[sk[0]-1, 0]), int(kpts[sk[0]-1, 1]))
                pos2 = (int(kpts[sk[1]-1, 0]), int(kpts[sk[1]-1, 1]))

                if use[sk[0]-1] and use[sk[1]-1]:
                    color = self.skeleton_link_colors[sk_id].tolist()
                    cv2.line(img, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord = int(kpt[0]), int(kpt[1])

                color = self.keypoint_colors[kid].tolist()
                if use[kid]:
                    cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           color, -1)
                    cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           (255, 255, 255))


def select_data():
    sel_data = {}
    # initialize clip model
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    for i, split in enumerate(splits):
        imgid2ann = {}
        cap_ann = COCO(os.path.join(coco_dir, captions_fls[i]))
        kp_ann = COCO(os.path.join(coco_dir, keypoints_fls[i]))
        
        # select images based on keypoints and crowd conditions
        for img_id in kp_ann.getImgIds():
            ann_ids = kp_ann.getAnnIds(imgIds=img_id)
            if len(ann_ids) == 1:
                ann = kp_ann.loadAnns(ann_ids)[0]   # single only single annotation is present
                if not ann["iscrowd"] and ann["num_keypoints"] >= int(17*sel_kp_frac):
                    del ann["segmentation"]
                    imgid2ann[img_id] = ann
        # gather captions for the select images
        for img_id in cap_ann.getImgIds():
            if img_id in imgid2ann.keys():
                ann_ids = cap_ann.getAnnIds(imgIds=img_id)
                anns = cap_ann.loadAnns(ann_ids)
                captions = []
                for ann in anns:
                    captions.append(ann["caption"])
                imgid2ann[img_id]["captions"] = captions
        # clip based caption selection
        for imgid, ann in tqdm(imgid2ann.items()):
            img_meta = cap_ann.imgs[imgid]
            img_pth = os.path.join(coco_dir, f"{split}2014", img_meta["file_name"])
            img = Image.open(img_pth).convert("RGB")
            processed_image = preprocess(img)
            processed_image = processed_image.unsqueeze(0).cuda()
            text_tokens = clip.tokenize([cap for cap in imgid2ann[imgid]["captions"]]).cuda()
            with torch.no_grad():
                image_features = model.encode_image(processed_image).float()
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            sel_idx = text_probs.argmax().item()
            sel_cap = imgid2ann[imgid]["captions"][sel_idx]
            imgid2ann[imgid]["caption"] = sel_cap
        # add image data
        for imgid, ann in imgid2ann.items():
            img_meta = cap_ann.imgs[imgid]
            ann.update(img_meta)

        sel_data[split] = imgid2ann
    
    # merge the splits
    train_len = len(sel_data["train"])
    val_len = len(sel_data["val"])
    sel_data["train"].update(sel_data["val"])
    del sel_data["val"]
    assert len(sel_data["train"]) == train_len + val_len
    # save
    with open(f"coco_single_person_dataset.json", "w") as fp:
        json.dump(sel_data, fp)


def generate_sk_control():
    # check op dir
    if not os.path.exists(KP_DIR):
        os.makedirs(KP_DIR)
    # generation settings
    cap_ann = COCO(os.path.join(coco_dir, keypoints_fls[0]))
    cat_info = cap_ann.loadCats(cap_ann.getCatIds())
    kp_colors = [clr for clr, sk in sorted(zip(colors, cat_info[0]["skeleton"]), key=lambda l: l[1][0]) ]
    kp_colors = np.array(kp_colors)
    metainfo = {"keypoint_id2name": {i: body_part for i, body_part in enumerate(cat_info[0]["keypoints"])},
                "keypoint_name2id": {body_part: i for i, body_part in enumerate(cat_info[0]["keypoints"])},
                "keypoint_colors": kp_colors,
                "skeleton_links": cat_info[0]["skeleton"],
                "skeleton_link_colors": colors}
    visualizer = FastVisualizer(metainfo)
    # read selected json
    with open(f"coco_single_person_dataset.json") as fp:
        sel_data = json.load(fp)
    # vis
    for split in ["train", "val"]:
        for imgid, sample in tqdm(sel_data[split].items(), desc="generating keypoint imgs"):
            fn = Path(sample["file_name"])
            kp_fn = f"{fn.stem}_kp{fn.suffix}"
            kp_fl = os.path.join(KP_DIR, kp_fn)
            kpts = np.array(sample["keypoints"]).reshape((17,3))
            kpts, status = kpts[:,:2], kpts[:,2]
            img = np.zeros((sample["height"], sample["width"], 3), dtype=np.int8)
            visualizer.draw_pose(img, np.expand_dims(kpts, 0), np.expand_dims(status, 0))   # since only single person per pic
            cv2.imwrite(kp_fl, img)


def sel_to_hf():
    with open("coco_single_person_dataset.json") as fp:
        data = json.load(fp)
    new_data = {"image": [],
            "text": [],
            "conditioning_image": []}
    for split in data.keys():
        for imgid, sample in data[split].items():
            img_pth = os.path.join(coco_dir, f"{split}2014", sample["file_name"])
            text = sample["caption"]
            fn = Path(sample["file_name"])
            conditioning_image = os.path.join(KP_DIR, f"{fn.stem}_kp{fn.suffix}")
            new_data["image"].append(img_pth)
            new_data["text"].append(text)
            new_data["conditioning_image"].append(conditioning_image)
    with open("coco14.json", "w") as fp:
        json.dump(new_data, fp)


if __name__ == "__main__":
    if SELECT_DATA:
        select_data()
    if GEN_KP_CTRL:
        generate_sk_control()
    if GEN_JF_JSON:
        sel_to_hf()
