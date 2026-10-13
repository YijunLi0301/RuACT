from io import BytesIO
from typing import Optional
import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))
    return fetch_image(image, image_patch_size=image_patch_size)


VIDEO_FORMAT_HELP = (
    "Currently, we only support the video formats introduced in qwen2-vl.\nRefer "
    'to \n\neg.\n{\n    "type": "video",\n    "video": [\n        "file:///path/to/fra'
    'me1.jpg",\n        "file:///path/to/frame2.jpg"\n    ]\n}\n\n{\n    "type": "vide'
    'o",\n    "video": "file:///path/to/video.mp4"\n}\n# Defaults to fps=2, min_fra'
    'mes=4, max_frames=768\n\n{\n    "type": "video",\n    "video": "file:///path/to'
    '/video.mp4",\n    "fps": 2,\n    "min_frames": 1,\n    "max_frames": 32\n}\n'
)


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor:
    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"
    video = dict(video)
    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames
    return fetch_video(
        video,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )


def process_multi_modal_inputs_for_minicpmo(
    input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
):
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device)
            - left_padding_length[i]
            + cu_seqlens[i]
        )
        image_bounds.append(image_bound)
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])
    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}
