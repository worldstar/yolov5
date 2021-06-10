import argparse
from os import name
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import base64
import json
import os
import shutil
from numpy import random, save

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# python yolov5DetectToJson.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source ./test/images --save-txt --save-conf

# add Data json
# image_dirPath: 照片存放路徑
# json_dirPath: json存放路徑
# imageName: 照片名稱
# nameFile: josn名稱
# path: 原始照片路徑
# image: 已讀取照片的記憶體位置
# shapes: label格式
def createLebelmeJson(image_dirPath, json_dirPath, imageName, nameFile, path, image, shapes):

    jsonFile = {}
    version = "4.5.7"
    flags = {}


    imagePath = os.path.join(os.path.dirname(__file__), ".{0}/{1}".format(image_dirPath[(image_dirPath.index("images") - 1):], imageName))
    imagePath = imagePath.replace("\\", "/")

    imageData = str(base64.b64encode(open(path, "rb").read()))
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    # create jsonFile format
    if shapes != []:
        jsonFile = {"version": version ,\
                    "flags": flags,\
                    "shapes": shapes,\
                    "imagePath": imagePath,\
                    "imageData": imageData[2:-1],\
                    "imageHeight": imageHeight,\
                    "imageWidth": imageWidth}

        # create Images
        shutil.copyfile(path, "{0}/{1}".format(image_dirPath.replace("\\", "/"), imageName))


        # create .json
        f = open(json_dirPath + "/" + nameFile, 'w')
        f.write(json.dumps(jsonFile, indent = 2))
        f.close()


# get Shapes
# shapes: label格式
# labelName: label名字
# positionXY: 座標
# group_id:
# shape_type:
# flags:
def getShapes(shapes, labelName, positionXY, group_id, shape_type, flags):

    points = []

    points.append([positionXY[0], positionXY[1]])
    points.append([positionXY[2], positionXY[3]])

    shapes.append({"label": labelName,\
                   "points": points,\
                   "group_id": group_id,\
                   "shape_type": shape_type,\
                   "flags": flags})

    return shapes

# Create dir
def createDirImage(image_dirPath):
    os.makedirs(image_dirPath)


# # cal ratio
# def calRatio(shapes):
    
#     w1 = 0
#     h1 = 0
    
#     w2 = 0
#     h2 = 0

#     for data in shapes:

#         # small (1)
#         if data['label'] == "cup":
#             w1 = data['points'][1][0]
#             h1 = data['points'][1][1]

#         # big   (2)
#         if data['label'] == "disk":
#             w2 = data['points'][1][0]
#             h2 = data['points'][1][1]
    
#     if w1 < w2 and h1 < h2:
#         ratio_w = w1 / w2
#         ratio_h = h1 / h2
#         print("")
#         print("w1: {0}, h1: {1}, w2: {2}, h2: {3}".format(w1, h1, w2, h2))
#         print("")
#         print("ratio_w: {0}, ratio_h: {1}".format(ratio_w, ratio_h))
#         print("")
#     else:
#         print("Width or height Error!")
    



def detect(save_img=False):

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    # 新增json資料夾
    createDirImage("{0}/images".format(save_dir))

    
    for path, img, im0s, vid_cap in dataset:

        # shapes Data
        shapes = []
        boxes = 0

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # shapes in data
                    labelName = ""
                    points = []
                    shape_type = "rectangle"

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        
                        boxes += 1

                        # create shapes format
                        getShapes(shapes, names[int(cls)], torch.tensor(xyxy).view(1, 4).view(-1).tolist(), None, "rectangle", {})

            # cal Ratio
            # if boxes == 2:
            #     calRatio(shapes)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # createLebelmeJson
            createLebelmeJson("{0}/images".format(save_dir), "{0}".format(save_dir).replace("\\", "/"), p.name[:-4] + ".JPG", p.name[:-4] + ".json", path, im0s, shapes)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
