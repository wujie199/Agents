# OCR 处理器安装说明

## 使用 conda py3.11 环境

### 1. 激活环境

```bash
conda activate py3.11
```

### 2. 安装 PaddlePaddle (CPU版本)

```bash
# CPU 版本
/opt/miniconda3/envs/py3.11/bin/pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 或 GPU 版本 (CUDA 11.8)
# /opt/miniconda3/envs/py3.11/bin/pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 3. 安装 PaddleOCR

```bash
/opt/miniconda3/envs/py3.11/bin/pip install paddleocr
```

### 4. 安装 PDF 处理依赖

```bash
# 安装 pdf2image
/opt/miniconda3/envs/py3.11/bin/pip install pdf2image

# 安装 poppler (系统依赖)
# macOS:
brew install poppler

# Ubuntu/Debian:
# sudo apt-get install poppler-utils

# Windows:
# 下载: https://github.com/oschwartz10612/poppler-windows/releases
# 解压后将 bin 目录添加到 PATH
```

### 5. 运行 OCR 处理器

```bash
# 使用 py3.11 环境运行
/opt/miniconda3/envs/py3.11/bin/python document/ocr/processor.py

# 或激活环境后运行
conda activate py3.11
python document/ocr/processor.py
```

## 快速安装（一键脚本）

```bash
# 激活环境
conda activate py3.11

# 安装所有 Python 依赖
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
pip install pdf2image

# 安装 poppler (macOS)
brew install poppler

# 运行
python document/ocr/processor.py
```

## 验证安装

```bash
# 验证 PaddlePaddle
python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"

# 验证 PaddleOCR
python -c "from paddleocr import PaddleOCR; print('PaddleOCR 安装成功')"

# 验证 pdf2image
python -c "from pdf2image import convert_from_path; print('pdf2image 安装成功')"
```

## 常见问题

### 1. PaddlePaddle 安装失败

```bash
# 尝试使用清华源
pip install paddlepaddle==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. poppler 未找到

```bash
# macOS
brew install poppler

# 验证
which pdfinfo  # 应该输出 /opt/homebrew/bin/pdfinfo 或类似路径
```

### 3. GPU 版本安装

如果有 NVIDIA GPU 且已安装 CUDA：

```bash
# CUDA 11.8
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.6
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

## 完整依赖列表

```
paddlepaddle==3.0.0    # PaddlePaddle 深度学习框架
paddleocr              # PaddleOCR OCR工具
pdf2image              # PDF转图片
poppler                # PDF处理工具（系统依赖）
```
