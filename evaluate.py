import os
import json
import hydra
import torch
import numpy as np
import cv2
import yaml  
import math

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

from models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from misc.util import add_source_frame_id
from misc.visualise_3d import save_ply

from datasets.util import create_datasets


def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def generate_video(model, cfg, dataloader, device=None, video_root_path= None, scene_ids = [0], original_video = False):
    model_model = get_model_instance(model)
    model_model.set_eval()
    dataloader_iter = iter(dataloader)
    scene_num = 0
    now = datetime.now()
    # Create a timestamped directory for video output
    timestamped_dir = Path(f"{video_root_path}/{now:%Y-%m-%d}_{now:%H-%M-%S}")
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    for k in tqdm([i for i in range(len(dataloader.dataset)  // cfg.data_loader.batch_size)]):  # len(dataloader)
        inputs = next(dataloader_iter)
        if scene_ids:
            if scene_num not in scene_ids:
                scene_num += 1
                continue

        video_path = str(timestamped_dir / f"{scene_num}.mp4")   # "video_{inputs[('frame_id', 0)]}.mp4"

        if original_video:
            src_frame = inputs[('color', 0, 0)]
        else:
            with torch.no_grad():
                if device is not None:
                    to_device(inputs, device)
                target_frame_ids = list(range(1, inputs[('total_frame_num', 0)]))
                inputs["target_frame_ids"] = target_frame_ids
                outputs = model(inputs)
            src_frame = outputs[('color_gauss', 0, 0)]

        height, width = src_frame.shape[2], src_frame.shape[3]
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        # Convert the source frame to numpy format for writing
        src_frame_np = src_frame[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255
        src_frame_np = src_frame_np.astype(np.uint8)
        # Convert from RGB to BGR for OpenCV
        src_frame_np = cv2.cvtColor(src_frame_np, cv2.COLOR_RGB2BGR)
        print(f"there are {inputs[('total_frame_num', 0)]} frames in video")
        mid_num = inputs[('total_frame_num', 0)] // 2 + 1
        ff_id = 1
        src_added = False

        try: 
            while ff_id < inputs[('total_frame_num', 0)]:
                if ff_id == mid_num and not src_added:
                    video.write(src_frame_np)
                    src_added = True
                    continue
                if original_video:
                    cur = inputs[('color', ff_id, 0)] #ground truth current image
                else:
                    cur = outputs[('color_gauss', ff_id, 0)] #predicted current image

                cur_frame = cur[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255
                cur_frame = cur_frame.astype(np.uint8)
                # Convert from RGB to BGR for OpenCV
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)

                video.write(cur_frame)

                ff_id += 1
        finally:
            video.release()

        print(f"Video {scene_num} saved at {video_path}")
        scene_num += 1
    
def evaluate(model, cfg, evaluator, dataloader, device=None, save_vis=False, output_path = None):
    model_model = get_model_instance(model)
    model_model.set_eval()
    
    score_dict = {}
    match cfg.dataset.name:
        case "pixelsplat" | "scannetpp":
            # Override the frame indices used for evaluation
            target_frame_ids = [1, 2, 3]
            eval_frames = ["src", "tgt5", "tgt10", "tgt_rand"]
            for fid, target_name in zip(add_source_frame_id(target_frame_ids), eval_frames):
                score_dict[fid] = {"ssim": [], "psnr": [], "lpips": [], "name": target_name}

    dataloader_iter = iter(dataloader)
    now = datetime.now()
    out_dir = output_path / f"{now:%Y-%m-%d}_{now:%H-%M-%S}"
    out_dir.mkdir(exist_ok=True)
    spliced_images_list = []
    for k in tqdm([i for i in range(len(dataloader.dataset)  // cfg.data_loader.batch_size)]):
        if save_vis:
            print(f"saving images to: {out_dir}")
            seq_name = dataloader.dataset._seq_keys[k]
            out_out_dir = out_dir / seq_name
            out_out_dir.mkdir(exist_ok=True)
            out_pred_dir = out_out_dir / "pred"
            out_pred_dir.mkdir(exist_ok=True)
            out_gt_dir = out_out_dir / "gt"
            out_gt_dir.mkdir(exist_ok=True)
            out_dir_ply = out_out_dir / "ply"
            out_dir_ply.mkdir(exist_ok=True)
            out_spliced_dir = out_out_dir / "spliced"
            out_spliced_dir.mkdir(exist_ok=True)

        try:
            inputs = next(dataloader_iter)
        ## not used ##
        except Exception as e:
            if cfg.dataset.name == "re10k" or cfg.dataset.name == "pixelsplat":
                if cfg.dataset.test_split in ["pixelsplat_ctx1", "pixelsplat_ctx2", "latentsplat_ctx1", "latentsplat_ctx2"]:
                    print(f"Failed to read example {k}")
                    continue
            raise e
        
        with torch.no_grad():
            if device is not None:
                to_device(inputs, device)
            inputs["target_frame_ids"] = target_frame_ids
            outputs = model(inputs)
            
        for f_id in score_dict.keys():
            pred = outputs[('color_gauss', f_id, 0)]
            gt = inputs[('color', f_id, 0)]  # Output is directly from input
            # Should work in for B>1, however, be careful of reduction
            out = evaluator(pred, gt)
            if save_vis:
                save_ply(outputs, out_dir_ply / f"{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
                pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
                gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
                
                # Save individual images
                plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
                plt.imsave(str(out_gt_dir / f"{f_id:03}.png"), gt)
                
                # Concatenate images horizontally (splicing)
                spliced_image = np.concatenate((gt, pred), axis=1)
                spliced_image_uint8 = (spliced_image * 255).astype(np.uint8)
                spliced_image_path = str(out_spliced_dir / f"{f_id:03}.png")
                plt.imsave(spliced_image_path, spliced_image_uint8)

                spliced_images_list.append(spliced_image_uint8)

            for metric_name, v in out.items():
                score_dict[f_id][metric_name].append(v)

        if spliced_images_list:
            total_image = np.vstack(spliced_images_list)  
            total_image_path = str(out_spliced_dir / f"inAll.png") 
            plt.imsave(total_image_path, total_image)
            spliced_images_list = []

    metric_names = ["psnr", "ssim", "lpips"]
    score_dict_by_name = {}
    for f_id in score_dict.keys():
        score_dict_by_name[score_dict[f_id]["name"]] = {}
        for metric_name in metric_names:
            # Compute mean
            score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
            # Original dict has frame ids as integers, for JSON out dict we want to change them
            # to the meaningful names stored in dict
            score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]

    result_content = []
    for metric in metric_names:
        vals = [score_dict_by_name[f_id][metric] for f_id in eval_frames]
        metric_line = f"{metric}: {np.mean(np.array(vals))}"
        print(metric_line)
        result_content.append(metric_line)

    json_content = json.dumps(score_dict_by_name, indent=4)

    # Write the combined content to the result.txt file
    result_txt_path = out_dir / "result.txt"
    with open(result_txt_path, "w") as txt_file:
        txt_file.write("\n".join(result_content) + "\n\n")
        txt_file.write(json_content)

    return score_dict_by_name

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)

    base_dir = Path(__file__).resolve().parent

    ckpt_path = base_dir / cfg.ckpt_path
    if ckpt_path.exists():
        model.load_model(ckpt_path, ckpt_ids=0)

    split = "test"
    save_vis = cfg.save_vis
    video_mode = cfg.video_mode
    original_video = cfg.original_video
    scene_ids = cfg.scene_ids
    output_path = base_dir / cfg.output_path    
    dataset, dataloader = create_datasets(cfg, split = split)

    if video_mode:
        generate_video(model, cfg, dataloader, device=device, video_root_path = output_path, scene_ids = scene_ids, original_video= original_video)
    else:
        evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
        evaluator.to(device)

        score_dict_by_name = evaluate(model, cfg, evaluator, dataloader, 
                                    device=device, save_vis=save_vis, output_path = output_path)
        print(json.dumps(score_dict_by_name, indent=4))
        if cfg.dataset.name=="re10k" or cfg.dataset.name=="pixelsplat":
            with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
                json.dump(score_dict_by_name, f, indent=4)
        with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    

if __name__ == "__main__":
    main()
