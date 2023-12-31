## 利用xtts克隆声音，微调模型

### 将一些微信聊天语音文件转换为mp3格式
借助silk-v3-decoder来批处理
```bash
#!/bin/bash

# 指定要遍历的文件夹路径
target_folder="/Users/jade_mayer/projects/tts/data/wechat_audios_of_lww_20231126"

# 查找特定后缀的文件并进行处理
find "$target_folder" -type f -name "*.silk" -print0 | while IFS= read -r -d '' file; do
    echo "处理文件: $file"
    # 在这里执行你的处理操作，比如读取文件内容或进行其他操作
	./converter.sh $file mp3
done

```

### clone xtts库并安装依赖
python版本要>=3.9:
```bash
conda create -n xtts python=3.10
```
```bash
conda activate xtts
```
安装TTS库及其依赖
```bash
./setup_xtts.sh
```
如遇到网络问题，可以开启终端代理：
```bash
export https_proxy=http://127.0.0.1:7891 http_proxy=http://127.0.0.1:7891 all_proxy=socks5://127.0.0.1:7891
```

### 下载whisper模型来处理audio数据

faster_whisper默认下载到cache文件夹
```bash
ls ~/.cache/huggingface/hub/models--openai--whisper-large-v2/
```
```python
# 下载whisper模型
import torch
from faster_whisper import WhisperModel
# Loading Whisper
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Loading Whisper Model!")
asr_model = WhisperModel("large-v2", device=device, compute_type="float16")
```
如果遇到网络问题，使用hf的方法下载后再利用faster_whisper加载
```python
from huggingface_hub import snapshot_download
from huggingface_hub import login
login()

model_id="guillaumekln/faster-whisper-large-v2"
snapshot_download(repo_id=model_id,
                 local_dir_use_symlinks=False, revision="main")
```
mac下用int8加载模型：
```python
# 加载whisper模型
import torch
from faster_whisper import WhisperModel
# Loading Whisper
device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Loading Whisper Model!")
asr_model = WhisperModel("large-v2", device=device, compute_type="int8", local_files_only=True)
```