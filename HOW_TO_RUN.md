# ACRE Crop–Weed: Detection → SAM2 → Segmentation

This document describes how to reproduce the pipeline used in the given task.

## 1. Environment setup
Clone git project:
```
git clone git@github.com:alexshade15/Detection_to_Segmentation_Pipeline.git Detection_to_Segmentation_Pipeline
cd Detection_to_Segmentation_Pipeline
```
Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
Install required python libs:
```
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics timm
pip install opencv-python numpy matplotlib
pip install huggingface_hub albumentations
pip install segmentation-models-pytorch
```

## 2. Download the ACRE dataset
```
wget https://zenodo.org/api/records/8102217/files-archive
unzip 8102217.zip -d data/
cd data/
unzip The_ACRE_Crop-Weed_Dataset.zip -d The_ACRE_Crop-Weed_Dataset/
cd ..
```

## 3. Evaluate dataset
```
Use ./src/DataAnalisys.ipynb to check dataset composition
```

## 4. Convert ACRE annotations → YOLO format
- classes: crop (0), weed (1)
- splits: train, val, test based on split_dictionary.json

Run the conversion script:
```
python3 src/convert_acre_yolo.py
```

## 5. Train YOLO11n
```
python3 src/Train_A.py
```
Save the best model to: runs_acre/yolo11n_crop_weed/weights/best.pt

## 6. Run YOLO inference and export detections to JSON
```
python3 src/run_inference.py
```
Output:
outputs/detections_partA/
  detections_yolo11n_train.json
  detections_yolo11n_val.json
  detections_yolo11n_test.json

## 7. Clone SAM2
```
mkdir sam2
cd sam2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ../..
```

## 8. Generate semantic masks with SAM2
```
python3 src/generate_sam_masks.py
```

## 9. Qualitative evaluation of SAM2 masks
```
python3 src/eval_SAM2_masks.py
```

## 10. Train the semantic segmentation model: UNet
```
python3 src/Train_B.py
```

## 11. Evaluation of Unet
```
python3 src/eval_seg.py
```




