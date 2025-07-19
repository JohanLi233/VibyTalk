import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm


def extract_audio(path, out_path, sample_rate=16000):
    print(f"[INFO] ===== extract audio from {path} to {out_path} =====")

    # 修复：使用更精确的ffmpeg参数确保时长匹配
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i",
        path,  # 输入文件
        "-vn",  # 不处理视频流
        "-acodec",
        "pcm_s16le",  # 指定音频编码器
        "-ar",
        str(sample_rate),  # 采样率
        "-ac",
        "1",  # 单声道
        "-async",
        "1",  # 音频同步，防止时长漂移
        "-avoid_negative_ts",
        "make_zero",  # 避免负时间戳
        out_path,
    ]

    print(f"[INFO] 执行命令: {' '.join(cmd)}")

    import subprocess

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("[INFO] ===== 音频提取成功 =====")
        else:
            print(f"[ERROR] 音频提取失败: {result.stderr}")
            raise RuntimeError(f"音频提取失败: {result.stderr}")

        # 验证提取的音频时长
        verify_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            out_path,
        ]
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        if verify_result.returncode == 0:
            import json

            info = json.loads(verify_result.stdout)
            duration = float(info.get("format", {}).get("duration", 0))
            print(f"[INFO] 提取的音频时长: {duration:.3f} 秒")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 音频提取失败: {e}")
        print(f"[ERROR] 错误输出: {e.stderr}")
        # 如果新方法失败，回退到原始方法
        print("[INFO] 回退到原始提取方法...")
        old_cmd = f"ffmpeg -y -i {path} -f wav -ar {sample_rate} {out_path}"
        os.system(old_cmd)

    print("[INFO] ===== extracted audio =====")


def extract_images(path):
    """
    从视频中提取图像帧。

    @param path 视频文件路径
    """
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    if not os.path.exists(full_body_dir):
        os.makedirs(full_body_dir, exist_ok=True)

    # 验证视频帧率
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps != 25:
        raise ValueError("Your video fps should be 25!!!")

    print(f"[INFO] 视频帧率: {fps} FPS，总帧数: {total_frames}")
    print(f"[INFO] 提取 {total_frames} 帧图像...")

    # 使用 FFmpeg 提取图像帧
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-vf",
        "fps=25",
        "-q:v",
        "2",
        "-start_number",
        "0",
        os.path.join(full_body_dir, "%d.jpg"),
    ]

    print(f"[INFO] 执行命令: {' '.join(extract_cmd)}")

    import subprocess

    # 使用进度条显示FFmpeg提取进度
    with tqdm(total=total_frames, desc="FFmpeg 提取帧", unit="帧") as pbar:
        try:
            process = subprocess.Popen(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )

            # 监控FFmpeg输出以更新进度条
            while True:
                output = process.stderr.readline() if process.stderr else ""
                if output == "" and process.poll() is not None:
                    break
                if output:
                    # 尝试从FFmpeg输出中解析帧数
                    if "frame=" in output:
                        try:
                            frame_part = output.split("frame=")[1].split()[0]
                            current_frame = int(frame_part)
                            pbar.n = min(current_frame, total_frames)
                            pbar.refresh()
                        except (ValueError, IndexError):
                            pass

            # 等待进程完成
            return_code = process.poll()
            if return_code == 0:
                pbar.n = total_frames
                pbar.refresh()
                print("[INFO] ===== 图像提取成功 =====")
            else:
                _, stderr = process.communicate()
                print(f"[ERROR] 图像提取失败: {stderr}")
                raise RuntimeError(f"图像提取失败: {stderr}")

            # 验证提取结果
            extracted_files = [
                f for f in os.listdir(full_body_dir) if f.endswith(".jpg")
            ]
            extracted_count = len(extracted_files)

            print(f"[INFO] FFmpeg 提取完成，成功提取 {extracted_count} 帧")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg 提取失败: {e}")
            raise RuntimeError(f"FFmpeg 提取失败: {e}")


def get_audio_feature(wav_path, mel_bins=80):
    """
    提取mel频谱图特征

    参数:
        wav_path: 音频文件路径
        mel_bins: mel滤波器组数量
    """
    print(f"正在提取mel特征，mel_bins={mel_bins}...")

    # 使用mel_extractor提取特征
    os.system(f"python ./data_utils/mel_extractor.py --wav {wav_path} --num_mel_bins {mel_bins}")

    # 生成的特征文件名
    target_file = wav_path.replace(".wav", f"_mel_{mel_bins}.npy")

    if os.path.exists(target_file):
        print(f"Mel特征已保存到: {target_file}")
    else:
        print(f"错误: 未找到生成的mel特征文件 {target_file}")

    return target_file


def write_landmarks_batch(results, img_names, landmarks_dir):
    """
    批量写入关键点文件
    """

    def write_single_landmark(args):
        result, img_name, landmarks_dir = args
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        if result is not None:
            pre_landmark, x1, y1 = result
            with open(lms_path, "w") as f:
                for p in pre_landmark:
                    x, y = p[0] + x1, p[1] + y1
                    f.write(f"{x} {y}\n")
            return img_name, True
        else:
            # 如果没有检测到人脸，创建一个空的 .lms 文件作为标记
            with open(lms_path, "w") as f:
                pass  # 创建一个空文件
            return img_name, False

    print("写入关键点文件...")

    # 使用线程池并行写入文件，添加进度条
    with ThreadPoolExecutor(max_workers=8) as executor:
        write_args = [
            (result, name, landmarks_dir) for result, name in zip(results, img_names)
        ]

        # 创建进度条
        completed = []
        with tqdm(total=len(write_args), desc="写入关键点", unit="文件") as pbar:
            futures = [
                executor.submit(write_single_landmark, arg) for arg in write_args
            ]
            for future in futures:
                result = future.result()
                completed.append(result)
                pbar.update(1)

                # 更新进度条状态
                success_count = sum(1 for _, success in completed if success)
                pbar.set_postfix({"成功": f"{success_count}/{len(completed)}"})

    success_count = sum(1 for _, success in completed if success)
    failed_count = len(completed) - success_count

    print(f"[INFO] 关键点文件写入完成: {success_count} 成功, {failed_count} 失败")

    if failed_count > 0:
        failed_files = [name for (name, success) in completed if not success]
        print(f"[WARNING] 以下文件处理失败: {failed_files[:10]}...")  # 只显示前10个


def get_landmark(path, landmarks_dir, batch_size=512):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")

    from get_landmark import Landmark

    # 获取所有图像文件
    img_files = [f for f in os.listdir(full_img_dir) if f.endswith(".jpg")]
    try:
        img_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        img_files.sort()

    if not img_files:
        print("[WARNING] No image files found!")
        return

    print(f"[INFO] Found {len(img_files)} images to process")

    # 创建关键点检测器
    landmark = Landmark(batch_size=batch_size)

    # 记录总时间和总成功数
    total_processing_time = 0
    total_success_count = 0

    # 手动分批处理并添加总进度条
    with tqdm(total=len(img_files), desc="整体关键点检测进度", unit="张") as pbar:
        for i in range(0, len(img_files), batch_size):
            batch_img_files = img_files[i : i + batch_size]
            batch_img_paths = [os.path.join(full_img_dir, f) for f in batch_img_files]

            # 记录当前批次开始时间
            start_time = time.time()

            # 批量检测关键点 (检测器内部有自己的批处理逻辑和进度条)
            results = landmark.batch_detect(batch_img_paths)

            # 立即写入当前批次的结果
            write_landmarks_batch(results, batch_img_files, landmarks_dir)
            
            # 更新处理时间和成功数
            end_time = time.time()
            total_processing_time += end_time - start_time
            total_success_count += sum(1 for r in results if r is not None)
            
            pbar.update(len(batch_img_files))

    # 计算并打印最终统计信息
    fps = len(img_files) / total_processing_time if total_processing_time > 0 else 0
    print(f"\n[INFO] Landmark detection completed in {total_processing_time:.2f}s")
    print(f"[INFO] Average processing speed: {fps:.2f} FPS")
    print(
        f"[INFO] Overall success rate: {total_success_count}/{len(img_files)} ({total_success_count / len(img_files) * 100:.1f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to video file")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size for landmark detection"
    )
    parser.add_argument(
        "--mel_bins",
        type=int,
        default=80,
        help="mel滤波器组数量，默认80",
    )
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, "aud.wav")
    landmarks_dir = os.path.join(base_dir, "landmarks")

    os.makedirs(landmarks_dir, exist_ok=True)

    print("=" * 60)
    print("🎵 STEP 1: 提取音频")
    print("=" * 60)
    extract_audio(opt.path, wav_path)

    print("\n" + "=" * 60)
    print("🖼️ STEP 2: 提取视频帧")
    print("=" * 60)
    extract_images(opt.path)

    print("\n" + "=" * 60)
    print("🎯 STEP 3: 检测人脸关键点")
    print("=" * 60)
    get_landmark(opt.path, landmarks_dir, opt.batch_size)

    print("\n" + "=" * 60)
    print(f"🔊 STEP 4: 提取音频特征 (使用 mel_bins={opt.mel_bins})")
    print("=" * 60)
    get_audio_feature(wav_path, opt.mel_bins)

    print("\n" + "=" * 60)
    print("✅ 处理完成!")
    print("=" * 60)
