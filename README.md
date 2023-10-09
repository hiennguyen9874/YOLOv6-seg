# Export YOLOv6-seg to ONNX and TensorRT

This implimentation is based on [YOLOv6](https://github.com/meituan/YOLOv6/tree/yolov6-seg).

## Install

- [TensorRT OSS Plugin](https://github.com/hiennguyen9874/TensorRT)

- [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

## Usage

### Download model

### Export with roi-align

#### Export ONNX

- `python3 deploy/ONNX-seg/export_onnx.py --weights ./weights/yolov6l_seg.pt --img-size 640 640 --batch-size 1 --simplify --dynamic-batch --end2end --topk-all 100 --iou-thres 0.65 --conf-thres 0.5 --device 1 --cleanup --mask-resolution 56 --opset 14 --roi-align`

- [scripts](notebooks/Yolov7onnx_mask-roialign.ipynb)

#### Export TensorRT

- `python3 deploy/ONNX-seg/export_onnx.py --weights ./weights/yolov6l_seg.pt --img-size 640 640 --batch-size 1 --simplify --dynamic-batch --end2end --topk-all 100 --iou-thres 0.65 --conf-thres 0.5 --device 1 --trt --cleanup --mask-resolution 56 --opset 14 --roi-align`

- `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov6l_seg.onnx --saveEngine=./weights/yolov6l_seg.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640 --shapes=images:1x3x640x640`

- [scripts](notebooks/YOLOv7trt_mask-roialign.ipynb)

### Export without roi-align

#### Export ONNX

- `python3 deploy/ONNX-seg/export_onnx.py --weights ./weights/yolov6l_seg.pt --img-size 640 640 --batch-size 1 --simplify --dynamic-batch --end2end --topk-all 100 --iou-thres 0.65 --conf-thres 0.5 --device 1 --cleanup --mask-resolution 56 --opset 14`

- [scripts](notebooks/Yolov7onnx_mask.ipynb)

#### Export TensorRT

- `python3 segment/export.py --data ./data/coco.yaml --weights ./weights/yolov7-seg.pt --batch-size 1 --device cpu --simplify --opset 14 --workspace 8 --iou-thres 0.65 --conf-thres 0.35 --include onnx --end2end --trt --cleanup --dynamic-batch`

- `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-seg.onnx --saveEngine=./weights/yolov7-seg-nms.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640 --shapes=images:1x3x640x640`

- [scripts](notebooks/YOLOv7trt_mask.ipynb)
