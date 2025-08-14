# YOLOHub

YOLO 模型資料前處理、標註轉換、自動標註、資料分割、訓練、推論與可視化的整合專案。

目的：快速建立物件偵測與語意分割實驗流程，並將繁瑣的 COCO/JSON/Mask 轉換、資料拆分、模型權重管理集中。

## 📂 目錄結構概覽

```
auto_annotation/        # SAM + YOLO 自動標註流程 
base_model/             # 放置原始預訓練 YOLO 權重 (建議自行下載)
coco2mask/              # COCO 轉二值 / mask 生成腳本
JSON2YOLO/              # JSON/COCO -> YOLO txt 轉換工具 (README 來自 Ultralytics 官方)
mask2yoloTXT/           # mask -> YOLO segmentation label 轉換
split/                  # 資料集自動切分工具
training/               # 訓練 YOLO 
use_model/              # 推論/驗證 
visual_ann/             # 標註可視化
extract_mask/           # 從標註 / 影像提取 mask 的流程
```

## ✅ 功能清單
- COCO JSON / Labelbox JSON 轉 YOLO txt (`JSON2YOLO`、`converter_coco.py`, `labelbox_json2yolo.py`)
- COCO -> mask 生成 (`coco2mask/`)
- mask -> YOLO segmentation label (`mask2yoloTXT/mask_convert_yolo_seg.py`)
- 自動標註 (SAM + YOLO 推論生成標註) (`auto_annotation/`)
- 資料集整合與自動切分 (`split/auto_split.ipynb`, `yolo_splitter.py`)
- yolo11 偵測 / 分割 訓練 notebook (`training/train_yolo11_det.ipynb`, `training/train_yolo11_seg.ipynb`)
- 推論與驗證 (`use_model/inference_seg.ipynb`, `model_validation/`)
- 標註/結果可視化 (`visual_ann/`)

## 🧩 主要腳本
| 路徑 | 說明 |
|------|------|
| `JSON2YOLO/general_json2yolo.py` | 一般 JSON -> YOLO | 
| `JSON2YOLO/converter_coco.py` | COCO 轉 YOLO | 
| `mask2yoloTXT/mask_convert_yolo_seg.py` | mask 轉 YOLO segmentation label |
| `coco2mask/coco2mask_generator.py` | 由 COCO 標註產生 mask 影像 |
| `split/yolo_splitter.py` | 資料集 train/val/test 分割 |

## 🛠 安裝環境
建議使用 Python 3.10+ (或與你現有環境相容版本)。

最少依賴 (轉換工具)：
```
pip install -r JSON2YOLO/requirements.txt
```
若使用 Ultralytics yolo11：
```
pip install ultralytics
```

額外常用：
```
pip install opencv-python pillow tqdm pyyaml numpy pandas
```

安裝 Pytorch (需確認 CUDA 版本)：

```
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

> 請先執行 `pip install ultralytics`，再執行 
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121`，以確保可使用 GPU 訓練。

## 📥 權重下載與放置
所有模型權重與訓練產物，請自行下載：

| 類型 | 建議來源 | 放置目錄 | 備註 |
|------|----------|----------|------|
| YOLO11 預訓練 | Ultralytics 官方 | `base_model/detection/` 或 `base_model/segmentation/` | [前往](base_model/README.md) | 
| 自訓練結果 | 你本地訓練輸出 | `training/*_training_result/<run_name>/weights/` | - |
| 推論用最佳權重 | 從訓練複製 | `use_model/<run_name>/weights/` | - |
| SAM / SAM2 權重 | Ultralytics 官方 | `auto_annotation/sam_model/` | [前往](https://docs.ultralytics.com/zh/models/sam) |

## ▶️ 使用流程建議
1. 準備原始標註 (COCO JSON / Labelbox JSON / masks)
2. 使用 `JSON2YOLO` 或 `mask2yoloTXT` 轉成 YOLO 格式
    - 若使用 `auto_annotation` 進行自動標記則跳過步驟 1、2。
3. 放入 `dataset_det/` 或 `dataset_seg/` 結構 (images / labels)
4. 用 `split/auto_split.ipynb` 進行 train/val/test 分割資料 
5. 執行 `training/train_yolo11_*.ipynb` 開始訓練
6. 完成後挑選最佳權重複製到 `use_model/<run_name>/weights/`
7. 使用 `use_model/inference_seg.ipynb` 或其他腳本推論
8. `visual_ann/` 進行結果檢視 / 驗證
9. 需要額外標註時用 `auto_annotation/` 生成

## 🧪 範例：COCO JSON -> YOLO
建議使用 `coco2mask.ipynb`
```
python JSON2YOLO/converter_coco.py \
  --json_dir path/to/coco/annotations \
  --save_dir dataset_det/yolo_data_integrated/labels
```
(確保對應影像放在 `dataset_det/yolo_data_integrated/images`)

---

