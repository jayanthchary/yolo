# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave
    webcam = source.isnumeric()

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "predictions.csv"

    if save_csv and not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "car", "motorcycle", "bus", "truck", "total"])

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    model.warmup(imgsz=(1, 3, *imgsz))
    seen = 0

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            frame = seen
            im0 = im0s[i].copy() if webcam else im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            counts = {
                "car": 0,
                "motorcycle": 0,
                "bus": 0,
                "truck": 0,
            }

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    if label in counts:
                        counts[label] += 1

                    if save_img or view_img:
                        annotator.box_label(
                            xyxy,
                            None if hide_labels else f"{label} {conf:.2f}",
                            color=colors(int(cls), True),
                        )

            if save_csv:
                total = sum(counts.values())
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame,
                        counts["car"],
                        counts["motorcycle"],
                        counts["bus"],
                        counts["truck"],
                        total,
                    ])

            im0 = annotator.result()
            if view_img:
                cv2.imshow("YOLOv5 Vehicle Detection", im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return

    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--save-csv", action="store_true")

    # show live detection window
    parser.add_argument("--view-img", action="store_true", help="show detection video")

    opt = parser.parse_args()

    # DO NOT REMOVE THESE
    print_args(vars(opt))
    return opt




def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
