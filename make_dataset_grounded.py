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
from copy import deepcopy
import cv2
from controlnet_aux import OpenposeDetector
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import concurrent.futures


# TODO
# 1. Right now the json still has both train and val splits, unify for more train data.

SELECT_DATA = True
GEN_KP_CTRL = True
GEN_GND_CTRL = True
GEN_JF_JSON = True


grounding_config_file = "../../final/GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
grounding_weight_file = "../../final/GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"


cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(grounding_config_file)
cfg.merge_from_list(["MODEL.WEIGHT", grounding_weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


OUT_DIR = "/scratch/rc5124/llvm/datasets/coco2014/sel_data"
coco_dir = "/scratch/rc5124/llvm/datasets/coco2014/"
KP_DIR = f"{OUT_DIR}/kp_imgs"
GND_DIR = f"{OUT_DIR}/gnd_imgs"
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
            if len(ann_ids) >= 1:
                sel_ann = []
                for ann in kp_ann.loadAnns(ann_ids):   # single only single annotation is present
                    if not ann["iscrowd"] and ann["num_keypoints"] >= int(17*sel_kp_frac):
                        del ann["segmentation"] # pass
                        sel_ann.append(ann)
                if len(sel_ann) >= 1:  # any image with more than 1 person with more than 30% kp visible.
                    imgid2ann[img_id] = {"sel_anns": sel_ann}
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
        limit_count = 0
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
            # limit
            limit_count += 1
            if limit_count > 550:
                break
        # add image data
        for imgid, ann in imgid2ann.items():
            img_meta = cap_ann.imgs[imgid]
            ann.update(img_meta)

        sel_data[split] = imgid2ann

    # merge the splits
    # train_len = len(sel_data["train"])
    # val_len = len(sel_data["val"])
    # sel_data["train"].update(sel_data["val"])
    # del sel_data["val"]
    # assert len(sel_data["train"]) == train_len + val_len
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
        for imgid, sample_ in tqdm(sel_data[split].items(), desc="generating keypoint imgs"):
            sample = sample_["sel_anns"]
            fn = Path(sample_["file_name"])
            kp_fn = f"{fn.stem}_kp{fn.suffix}"
            kp_fl = os.path.join(KP_DIR, kp_fn)
            img = np.zeros((sample_["height"], sample_["width"], 3), dtype=np.int8)
            for skel in sample:
                kpts = np.array(skel["keypoints"]).reshape((17,3))
                kpts, status = kpts[:,:2], kpts[:,2]
                visualizer.draw_pose(img, np.expand_dims(kpts, 0), np.expand_dims(status, 0))   # since only single person per pic
            cv2.imwrite(kp_fl, img)
            sample_["conditioning_image"] = kp_fl
    # save
    with open(f"coco_single_person_dataset.json", "w") as fp:
        json.dump(sel_data, fp)


def generate_sk_control_openpose():
    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    open_pose.to("cuda")
    # check op dir
    if not os.path.exists(KP_DIR):
        os.makedirs(KP_DIR)
    # read selected json
    with open(f"coco_single_person_dataset.json") as fp:
        sel_data = json.load(fp)
    # vis
    for split in ["train", "val"]:
        for imgid, sample in tqdm(sel_data[split].items(), desc="generating keypoint imgs"):
            fn = Path(sample["file_name"])
            kp_fn = f"{fn.stem}_kp{fn.suffix}"
            kp_fl = os.path.join(KP_DIR, kp_fn)
            img_pth = os.path.join(coco_dir, f"{split}2014", sample["file_name"])
            img = Image.open(img_pth).convert("RGB")
            kp_img = open_pose(img, hand_and_face=False)
            kp_img.save(kp_fl)
            sample["conditioning_image"].append(kp_fl)
    # write the json
    with open(f"coco_single_person_dataset.json", "w") as fp:
        json.dump(sel_data, fp)


def fraction(thresh = 500):
    # read selected json
    with open(f"coco_single_person_dataset.json") as fp:
        sel_data = json.load(fp)
    new_data = {"train": {},
                "val": {}}
    #
    for split in ["train", "val"]:
        imgid2samples = dict(list(sel_data[split].items())[:thresh])
        new_data[split] = imgid2samples
    # save
    with open(f"coco_single_person_dataset.json", "w") as fp:
        json.dump(new_data, fp)


def generate_grounding():
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    # check op dir
    if not os.path.exists(GND_DIR):
        os.makedirs(GND_DIR)
    # read selected json
    with open(f"coco_single_person_dataset.json") as fp:
        sel_data = json.load(fp)
    # generate grounding entities and bboxes
    for split in ["train", "val"]:
        for imgid, sample_ in tqdm(sel_data[split].items(), desc="generating grounding data"):
            sample = sample_["sel_anns"]
            img_pth = os.path.join(coco_dir, f"{split}2014", sample_["file_name"])
            pil_image = Image.open(img_pth).convert("RGB")
            image = np.array(pil_image)[:, :, [2, 1, 0]]
            caption = sample_["caption"]
            vis, pred = glip_demo.run_on_web_image(image, caption, 0.5)
            vis = vis[:, :, [2, 1, 0]]
            entities = []
            # save visualization
            fn = Path(sample_["file_name"])
            gnd_fn = f"{fn.stem}_gnd{fn.suffix}"
            gnd_fl = os.path.join(GND_DIR, gnd_fn)
            cv2.imwrite(gnd_fl, vis)
            # collect grounding
            lbls = pred.get_field("labels")
            if glip_demo.entities and glip_demo.plus:
                for i in lbls:
                    if i <= len(glip_demo.entities):
                        entities.append(glip_demo.entities[i - glip_demo.plus])
                    else:
                        entities.append('object')
                        print("generic entity name assigned - object")
            else:
                entities = ['object' for i in lbls]
                print("generic entity name assigned - object")
            boxes = pred.bbox.detach().cpu().numpy()
            h,w,_ = image.shape
            new_bboxes = deepcopy(boxes)
            new_bboxes[:, 0] = new_bboxes[:, 0]/w
            new_bboxes[:, 1] = new_bboxes[:, 1]/h
            new_bboxes[:, 2] = new_bboxes[:, 2]/w
            new_bboxes[:, 3] = new_bboxes[:, 3]/h
            new_bboxes = new_bboxes.tolist()
            sample_["grounding_nouns"] = entities
            sample_["grounding_bboxes"] = new_bboxes
    # save
    with open(f"coco_single_person_dataset.json", "w") as fp:
        json.dump(sel_data, fp)


def sel_to_hf():
    with open("coco_single_person_dataset.json") as fp:
        data = json.load(fp)
    new_data = {"image": [],
            "text": [],
            "conditioning_image": [],
            "grounding":[]}
    for split in data.keys():
        for imgid, sample_ in data[split].items():
            sample = sample_["sel_anns"]
            grounding_data = {"bbox": [], "noun": []}
            img_pth = os.path.join(coco_dir, f"{split}2014", sample_["file_name"])
            text = sample_["caption"]
            conditioning_image = sample_["conditioning_image"]
            new_data["image"].append(img_pth)
            new_data["text"].append(text)
            new_data["conditioning_image"].append(conditioning_image)
            grounding_data["noun"].append(sample_["grounding_nouns"])
            grounding_data["bbox"].append(sample_["grounding_bboxes"])
            new_data["grounding"].append(grounding_data)
    with open("coco14.json", "w") as fp:
        json.dump(new_data, fp)


if __name__ == "__main__":
    if SELECT_DATA:
        select_data()
    fraction()
    if GEN_KP_CTRL:
        generate_sk_control()
    if GEN_GND_CTRL:
        generate_grounding()
    if GEN_JF_JSON:
        sel_to_hf()
