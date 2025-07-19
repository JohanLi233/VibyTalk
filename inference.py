import argparse
import os
import cv2
import torch
import numpy as np
from unet import UNet
from tqdm import tqdm
from config import get_config, list_available_models, get_border_from_crop_size

parser = argparse.ArgumentParser(
    description="Inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--dataset", type=str, default="")
parser.add_argument(
    "--audio_feat",
    type=str,
    default="",
    help="Path to audio features file (if not provided, will auto-detect based on model_size)",
)
parser.add_argument("--save_path", type=str, default="")  # end with .mp4 please
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument(
    "--model_size",
    type=str,
    default="medium",
    choices=list_available_models(),
    help="Model size to use.",
)
parser.add_argument(
    "--use_onnx",
    action="store_true",
    help="Use ONNX model for inference instead of PyTorch",
)
parser.add_argument(
    "--onnx_model",
    type=str,
    default="model.onnx",
    help="Path to ONNX model file (default: model.onnx)",
)
args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
model_size = args.model_size
use_onnx = args.use_onnx
onnx_model_path = args.onnx_model

device = "cuda" if torch.cuda.is_available() else "mps"

# 获取模型配置
config = get_config(model_size)
print(f"使用模型配置: {model_size}")
print(f"输入分辨率: {config.input_resolution}")
print(f"谱频段数: {config.spectral_bins}")
if use_onnx:
    print(f"使用 ONNX 推理，模型文件: {onnx_model_path}")
else:
    print(f"使用 PyTorch 推理，检查点文件: {checkpoint}")

# 如果没有提供音频特征文件路径，则使用mel特征
if not audio_feat_path:
    audio_feat_path = os.path.join(dataset_dir, f"aud_mel_{config.spectral_bins}.npy")
    print(f"自动选择mel音频特征文件: {audio_feat_path}")

if not os.path.exists(audio_feat_path):
    raise FileNotFoundError(f"音频特征文件不存在: {audio_feat_path}")

print(f"使用音频特征文件: {audio_feat_path}")


def get_audio_features(features, index):
    """
    获取音频特征的滑动窗口。
    - 使用配置中定义的序列长度
    - 返回 (T, G, D) 形状的张量，不压平特征
    """
    # 1. 定义窗口大小 (中心点前后各 T/2)
    win_size_half = config.temporal_window_length // 2
    left = index - win_size_half
    right = index + win_size_half

    # 2. 处理边界情况，计算需要填充的量
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]

    # 3. 获取窗口内的特征
    #    形状为 (window_size, G, D)
    auds = features[left:right]

    # 4. 对窗口边界进行填充 (使用复制第一个/最后一个元素的方式)
    if pad_left > 0:
        left_pad = np.repeat(auds[0:1], pad_left, axis=0)
        auds = np.concatenate([left_pad, auds], axis=0)
    if pad_right > 0:
        right_pad = np.repeat(auds[-1:], pad_right, axis=0)
        auds = np.concatenate([auds, right_pad], axis=0)

    # 5. 直接返回 (T, G, D) 形状的张量
    return torch.from_numpy(auds.astype(np.float32))


audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir + "0.jpg")
h, w = exm_img.shape[:2]

# 修复 cv2.VideoWriter_fourcc 错误
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
video_writer = cv2.VideoWriter(save_path, fourcc, 25, (w, h))
step_stride = 0
img_idx = 0

# 初始化模型
if use_onnx:
    # 初始化 ONNX 模型
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "要使用 ONNX 推理，请安装 onnxruntime: `pip install onnxruntime`"
        )

    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")

    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )
    net = None  # 不需要 PyTorch 模型
else:
    # 初始化 PyTorch 模型
    net = UNet(6, model_size=model_size).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.eval()
    ort_session = None  # 不需要 ONNX 会话

for i in tqdm(range(audio_feats.shape[0]), desc="Inference"):
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx) + ".jpg"
    lms_path = lms_dir + str(img_idx) + ".lms"

    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_size = config.input_resolution
    crop_img = cv2.resize(
        crop_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA
    )
    # Convert from BGR to RGB for model processing
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img_ori = crop_img.copy()

    # 动态计算边界，保持比例
    border = get_border_from_crop_size(crop_size)
    inner_size = crop_size - 2 * border
    img_real_ex = crop_img[
        border : border + inner_size, border : border + inner_size
    ].copy()
    img_real_ex_ori = img_real_ex.copy()

    # 创建嘴部遮罩区域 (基于inner_size动态计算)
    mask_height = int(inner_size * 0.4)  # 遮罩高度为内部区域的40%
    mask_width = int(inner_size * 0.6)   # 遮罩宽度为内部区域的60%
    y_center = int(inner_size * 0.75)    # 遮罩中心Y位置为内部区域的75%
    x_center = inner_size // 2           # 遮罩中心X位置为内部区域的50%
    
    x1 = x_center - mask_width // 2
    y1 = y_center - mask_height // 2
    x2 = x_center + mask_width // 2
    y2 = y_center + mask_height // 2
    
    img_masked = cv2.rectangle(img_real_ex_ori, (x1, y1), (x2, y2), (0, 0, 0), -1)

    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0

    img_real_ex_T = torch.from_numpy(img_real_ex)
    img_masked_T = torch.from_numpy(img_masked)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None]

    audio_feat = get_audio_features(audio_feats, i)
    # 新的音频特征格式为 (T, G, D), e.g. (32, 4, 80)
    # 增加batch维度以适配模型输入 (B, T, G, D)
    audio_feat = audio_feat.unsqueeze(0)  # (1, T, G, D)

    # 推理
    if use_onnx:
        if ort_session is None:
            raise RuntimeError("ONNX 会话 (ort_session) 未初始化，请检查模型加载。")
        # ONNX 推理
        try:
            ort_inputs = {
                ort_session.get_inputs()[0].name: img_concat_T.cpu().numpy(),
                ort_session.get_inputs()[1].name: audio_feat.cpu().numpy(),
            }
            ort_outs = np.array(ort_session.run(None, ort_inputs))
            pred = ort_outs[0][0]  # 获取第一个输出的第一个batch
        except AttributeError as e:
            raise RuntimeError("ONNX 推理失败，可能未正确加载 ONNX 模型。") from e
        except Exception as e:
            raise RuntimeError(f"ONNX 推理过程中发生错误: {e}") from e
    else:
        if net is None:
            raise RuntimeError("PyTorch 模型 (net) 未初始化，请检查模型加载。")
        # PyTorch 推理
        audio_feat = audio_feat.to(device)
        img_concat_T = img_concat_T.to(device)

        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
        pred = pred[0].cpu().numpy()

    pred = pred.transpose(1, 2, 0) * 255
    pred = np.array(pred, dtype=np.uint8)

    # --- Start of fix ---
    # To mitigate boundary artifacts from the new model, we blend the predicted patch
    # with the original crop instead of a hard paste.
    # 1. Create a canvas with the predicted patch pasted.
    pasted_pred = crop_img_ori.copy()
    pasted_pred[border : border + inner_size, border : border + inner_size] = pred

    # 2. Create a blending mask with feathered edges.
    mask = np.zeros((crop_size, crop_size, 1), dtype=np.float32)
    mask[border : border + inner_size, border : border + inner_size] = 1.0
    # The blur kernel size is based on the border size
    # to create a smooth feathering effect (2 * border + 1).
    kernel_size = 2 * border + 1
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)[:, :, np.newaxis]

    # 3. Blend the pasted result with the original crop.
    crop_img_ori = (
        pasted_pred.astype(np.float32) * mask
        + crop_img_ori.astype(np.float32) * (1 - mask)
    ).astype(np.uint8)
    # --- End of fix ---

    # Convert back from RGB to BGR for video output
    crop_img_ori = cv2.cvtColor(crop_img_ori, cv2.COLOR_RGB2BGR)
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)
video_writer.release()

# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
