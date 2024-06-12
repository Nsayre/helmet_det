# V5 tests
# python ./yolov5/val.py --img 640 --batch 64 --data helmet_det_v5.yaml --weights model_results/YOLO_v5n_25epoch/weights/best.pt --task test
# V8 tests
# yolo val model=/home/nick/code/Nsayre/helmet_det/model_results/YOLO_v8n_25epoch/weights/best.pt data=helmet_det_v8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=cpu

# yolo val model=/home/nick/code/Nsayre/helmet_det/model_results/YOLO_v8n_50epoch/weights/best.pt data=helmet_det_v8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=cpu

# yolo val model=/home/nick/code/Nsayre/helmet_det/model_results/YOLO_v8s_50epoch/weights/best.pt data=helmet_det_v8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=cpu

# yolo val model=/home/nick/code/Nsayre/helmet_det/model_results/YOLO_v8m_25epoch/weights/best.pt data=helmet_det_v8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=cpu
