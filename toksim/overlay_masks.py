from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import os
from PIL import Image

def overlay_with_moviepy(video_path, mask_folder, output_path, fps=24, max_frames=None):
    clip = VideoFileClip(video_path)
    w, h = clip.size

    mask_files = sorted(os.listdir(mask_folder))
    if max_frames is not None:
        mask_files = mask_files[:max_frames]
        clip = clip.subclip(0, max_frames / clip.fps)

    mask_images = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask_np = np.array(Image.open(mask_path).convert("L").resize((w, h)))

        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[:, :, 1] = mask_np
        mask_images.append(colored_mask)

    mask_clip = ImageSequenceClip(mask_images, fps=clip.fps)

    def overlay(get_frame, t):
        frame = get_frame(t)
        mask_frame = mask_clip.get_frame(t)
        return (0.7 * frame + 0.3 * mask_frame).astype(np.uint8)

    final = clip.fl(overlay)
    final.write_videofile(output_path, codec="libx264", fps=fps, audio_codec="aac")


video_path = "/home/sxk230060/object_wiper/assets/video_eg_2/zebra_1_1_gt.mp4"
mask_folder = "/home/sxk230060/object_wiper/assets/video_eg_2/mask"
output_path = "/home/sxk230060/object_wiper/assets/video_eg_2/zebra_1_1_gt_overlay.mp4"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

overlay_with_moviepy(video_path, mask_folder, output_path, max_frames=24)