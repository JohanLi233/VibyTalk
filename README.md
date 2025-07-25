# VibyTalk

## Demo

### Video Demo
#### Direct inference in web browser
https://github.com/user-attachments/assets/c8746006-6dcc-41b1-a3d3-bdd25530459c


## Quick Start

### 0. Install

Install UV then run

```bash
uv sync
```

### 1. Prepare Video Data
Record a 3-minute video of yourself speaking clearly. Ensure good lighting and audio quality or use synthesized data.

### 2. Process Data

Put this video to a new dir.

```bash
python data_utils/process.py /path/to/new_dir
```

### 3. Train Model

```bash
python train.py --dataset_dir /path/to/new_dir --model_size nano --save_dir ./checkpoint
```

### 4. Export Model

```bash
python export_onnx.py --checkpoint ./checkpoints/nano_300.pth --output model.onnx --model_size nano
```

### 5. Deploy

#### Real-time Mode

```bash
python realtime.py --dataset /path/to/new_dir --wav_path ./processed_data/aud.wav --onnx_model model.onnx --model_size nano
```

#### Extract Dataset for Web

```bash
python extract_dataset_data.py --dataset /path/to/new_dir --model_size nano
```

#### Web Interface

```bash
cd web
pnpm install
pnpm run dev
```

