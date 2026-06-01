# OCR 处理器

串联版面分析模型和文字识别模型的 OCR 处理工具，**支持图片和PDF文档**。

## 功能特性

- ✅ 支持图片格式：PNG、JPG、JPEG、BMP 等
- ✅ 支持PDF文档：自动分页处理
- ✅ 版面分析：识别文档结构、检测文本区域
- ✅ 文字识别：高精度文本识别（平均准确率 84.01%）
- ✅ 批量处理：支持多图片/多PDF处理
- ✅ GPU加速：支持GPU推理加速

## 模型说明

### 1. 版面分析模型 (ansisy/PP-DocLayoutV3)

- **功能**: 文档版面分析，识别文档结构
- **能力**:
  - 检测文本区域位置（支持倾斜、弯曲表面）
  - 识别多种布局元素（标题、段落、表格、图表等）
  - 输出逻辑阅读顺序
- **支持类型**:
  - text (文本)
  - table (表格)
  - figure_title (图片标题)
  - doc_title (文档标题)
  - content (内容)
  - reference (参考文献)
  - 等 20+ 种类型

### 2. 文字识别模型 (ocr_v5/PP-OCRv5)

- **功能**: 文本行识别
- **能力**:
  - 支持简体中文、繁体中文、英文、日文
  - 支持手写文本识别
  - 支持竖排文本、拼音、生僻字
- **准确率**: 平均准确率 84.01%

## 安装依赖

```bash
# 安装 PaddlePaddle (CPU版本)
python -m pip install paddlepaddle==3.0.0

# 安装 PaddlePaddle (GPU版本，CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 安装 PaddleOCR
python -m pip install paddleocr
```

## 使用方法

### Python API

```python
from document.ocr.processor import OCRProcessor

# 创建处理器
processor = OCRProcessor(
    layout_model_dir="model/ansisy",    # 版面分析模型目录
    ocr_model_dir="model/ocr_v5",        # 文字识别模型目录
    use_gpu=False                        # 是否使用GPU
)

# 处理单张图片
result = processor.process("document.png")

# 获取结果
print(f"识别到的文本区域数量: {len(result.regions)}")
print(f"完整文本:\n{result.full_text}")

# 保存结果
processor.save_result(result, "output.json", format="json")
processor.save_result(result, "output.txt", format="txt")
```

### 仅使用版面分析

```python
result = processor.process(
    "document.png",
    use_layout=True,   # 使用版面分析
    use_ocr=False      # 不使用文字识别
)
```

### 仅使用文字识别

```python
result = processor.process(
    "document.png",
    use_layout=False,  # 不使用版面分析
    use_ocr=True       # 使用文字识别
)
```

### 处理PDF文档

```python
# 自动识别PDF，逐页处理
result = processor.process("document.pdf")

# 查看每页结果
for page in result.pages:
    print(f"第{page.page_number}页:")
    print(f"  区域数: {len(page.regions)}")
    print(f"  文本: {page.full_text[:100]}...")

# 获取完整文本（所有页拼接）
print(result.full_text)

# 指定PDF转图片的DPI（默认200，值越大越清晰但速度越慢）
result = processor.process("document.pdf", pdf_dpi=300)
```

### PDF处理说明

**依赖安装**：
```bash
# 安装 pdf2image
pip install pdf2image

# 安装 poppler（必需）
# macOS:
brew install poppler

# Ubuntu:
sudo apt-get install poppler-utils

# Windows: 
# 下载 poppler: https://github.com/oschwartz10612/poppler-windows/releases
# 解压后将 bin 目录添加到 PATH
```

**工作流程**：
```
PDF输入
    ↓
转换为图片（pdf2image）
    ↓
逐页进行OCR处理
    ↓
合并所有页结果
```

**DPI选择建议**：
- **150 DPI**: 快速处理，适合清晰文档
- **200 DPI**: 默认值，平衡质量和速度
- **300 DPI**: 高质量，适合模糊文档或小字
- **600 DPI**: 最高质量，处理时间较长

### 批量处理

```python
image_paths = ["doc1.png", "doc2.png", "doc3.png"]
results = processor.process_batch(image_paths)

for result in results:
    print(f"{result.image_path}: {len(result.regions)} 个区域")
```

### 命令行使用

```bash
# 处理图片
python document/ocr/processor.py document.png

# 处理PDF（自动识别）
python document/ocr/processor.py document.pdf

# 指定模型目录
python document/ocr/processor.py document.pdf \
    --layout-model model/ansisy \
    --ocr-model model/ocr_v5

# 使用GPU
python document/ocr/processor.py document.pdf --gpu

# 指定PDF DPI
python document/ocr/processor.py document.pdf --dpi 300

# 保存结果
python document/ocr/processor.py document.pdf -o output.json -f json

# 仅使用文字识别（跳过版面分析）
python document/ocr/processor.py document.pdf --no-layout
```

## 结果格式

### 图片结果格式

```json
{
  "image_path": "document.png",
  "regions": [
    {
      "region_type": "text",
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "confidence": 0.95,
      "content": "识别的文本内容"
    }
  ],
  "full_text": "完整的识别文本\n第二行...",
  "page_number": null,
  "metadata": {
    "use_layout": true,
    "use_ocr": true,
    "layout_model": "PP-DocLayoutV3",
    "ocr_model": "PP-OCRv5_server_rec",
    "layout_regions_count": 10
  }
}
```

### PDF结果格式

```json
{
  "pdf_path": "document.pdf",
  "pages": [
    {
      "image_path": "/tmp/ocr_pdf_xxx/page_1.png",
      "regions": [...],
      "full_text": "第一页文本内容",
      "page_number": 1,
      "metadata": {...}
    },
    {
      "image_path": "/tmp/ocr_pdf_xxx/page_2.png",
      "regions": [...],
      "full_text": "第二页文本内容",
      "page_number": 2,
      "metadata": {...}
    }
  ],
  "total_pages": 2,
  "full_text": "=== 第 1 页 ===\n第一页文本内容\n\n=== 第 2 页 ===\n第二页文本内容",
  "metadata": {
    "use_layout": true,
    "use_ocr": true,
    "dpi": 200,
    "layout_model": "PP-DocLayoutV3",
    "ocr_model": "PP-OCRv5_server_rec"
  }
}
    "layout_model": "PP-DocLayoutV3",
    "ocr_model": "PP-OCRv5_server_rec",
    "layout_regions_count": 10
  }
}
```

### 字段说明

- `image_path`: 输入图片路径
- `regions`: 识别区域列表
  - `region_type`: 区域类型（text/table/title等）
  - `bbox`: 边界框坐标（4个顶点）
  - `confidence`: 置信度（0-1）
  - `content`: 识别的文本内容
- `full_text`: 完整文本（所有区域拼接）
- `metadata`: 元数据信息

## 工作流程

```
图片输入
    ↓
版面分析模型
    ↓
检测文本区域位置
    ↓
文字识别模型 (ocr_v5)
    ↓
识别每个区域的文本
    ↓
返回结构化结果
```

## 性能优化

### 使用 GPU

```python
processor = OCRProcessor(use_gpu=True)
```

### 批量处理

批量处理可以减少模型加载时间：

```python
results = processor.process_batch(image_paths)
```

### 选择性使用

如果只需要文本内容，可以跳过版面分析：

```python
result = processor.process(image_path, use_layout=False, use_ocr=True)
```

## 注意事项

1. **模型加载**: 首次调用时会加载模型，可能需要几秒钟
2. **GPU 内存**: 使用 GPU 时注意内存占用，批量处理时建议分批
3. **图片格式**: 支持 PNG、JPG、JPEG、BMP 等常见格式
4. **倾斜文档**: 版面分析模型支持倾斜、弯曲文档的识别
5. **依赖安装**: 确保正确安装 PaddlePaddle 和 PaddleOCR

## 示例场景

### 场景1: 文档数字化

```python
processor = create_processor(use_gpu=True)
result = processor.process("scanned_doc.png")
processor.save_result(result, "output.txt", format="txt")
```

### 场景2: 表格识别

```python
result = processor.process("table.png")
table_regions = [r for r in result.regions if r.region_type == "table"]
```

### 场景3: 批量文档处理

```python
from pathlib import Path

processor = create_processor(use_gpu=True)
image_dir = Path("documents")
images = list(image_dir.glob("*.png"))

results = processor.process_batch([str(img) for img in images])

for result in results:
    output_name = Path(result.image_path).stem
    processor.save_result(result, f"output/{output_name}.json")
```
