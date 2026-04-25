import os
import torch
import torch.nn.functional as F
import torchvision.io as io

from toksim_utils import compute_token_score_per_video, get_models

def get_resized_tensors(x, H, W, F_out, mode="bilinear"):
    x = x[:F_out].permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(H, W), mode=mode)
    return x.permute(0, 2, 3, 1)

def read_mask_frames(frames_dir):
    frame_files = sorted(os.listdir(frames_dir))
    frames = []
    for fname in frame_files:
        img = io.read_image(os.path.join(frames_dir, fname))
        img = img.permute(1, 2, 0)
        frames.append(img)
    return torch.stack(frames, dim=0)

def get_token_score(video_path, gt_path, mask_dir, processor, model, device="cuda", max_frames=24):
    pred_vid, _, _ = io.read_video(video_path, pts_unit='sec')
    gt_vid, _, _ = io.read_video(gt_path, pts_unit='sec')
    mask_vid = read_mask_frames(mask_dir)

    F = min(mask_vid.size(0), gt_vid.size(0), pred_vid.size(0), max_frames)
    H = gt_vid.size(1)
    W = gt_vid.size(2)

    pred_vid = get_resized_tensors(pred_vid, H, W, F)
    gt_vid = get_resized_tensors(gt_vid, H, W, F)
    mask_vid = get_resized_tensors(mask_vid, H, W, F, mode="nearest")

    token_score_video, all_metrics = compute_token_score_per_video(pred_vid, gt_vid, mask_vid, \
                                                      processor, model, device=device)
    return token_score_video, all_metrics


def main():
    device = "cuda:0"
    processor, model = get_models(device=device)

    video_path = "/home/sxk230060/object_wiper/assets/video_eg_1/sphere_1_1_pred1.mp4"
    gt_path = "/home/sxk230060/object_wiper/assets/video_eg_1/sphere_1_1_gt.mp4"
    mask_dir = "/home/sxk230060/object_wiper/assets/video_eg_1/mask"

    curr_vid_score, curr_vid_metrics = get_token_score(video_path, gt_path, mask_dir, processor, model, device=device)

    curr_vid_score = curr_vid_score * 100

    print(f"Token score: {curr_vid_score:.2f}")
    # print(f"Metrics: {curr_vid_metrics}")

if __name__ == "__main__":
    main()
