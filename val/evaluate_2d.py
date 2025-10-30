import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 2D detections (Precision/Recall/AP/mAP)')
    parser.add_argument('--pred_json', type=str, required=True, help='Path to predictions JSON (from inference.py)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to KITTI label_2 directory')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold (default 0.5)')
    parser.add_argument('--classes', type=str, nargs='*', default=None, help='Optional class names to evaluate')
    parser.add_argument('--img_ext', type=str, default='.png', help='Image extension used in predictions (default .png)')
    return parser.parse_args()


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, ay2 - ay1 + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_predictions(pred_json):
    with open(pred_json, 'r') as f:
        preds = json.load(f)

    # 🟢 Normalize class names to match KITTI GT labels (e.g., Car, Pedestrian, Cyclist)
    for pred in preds:
        if "class_name" in pred and isinstance(pred["class_name"], str):
            pred["class_name"] = pred["class_name"].capitalize()

    per_image = defaultdict(list)
    class_names = set()
    for p in preds:
        image = Path(p['image']).name  # normalize filename only
        bbox = p['bbox']  # [x1, y1, x2, y2]
        score = float(p['score']) if p.get('score') is not None else 1.0
        cls_name = p.get('class_name') or str(p.get('class_id', 'unknown'))
        per_image[image].append({'bbox': bbox, 'score': score, 'cls': cls_name})
        class_names.add(cls_name)
    return per_image, sorted(class_names)


def parse_kitti_label_line(line):
    parts = line.strip().split(' ')
    cls = parts[0]
    if cls == 'DontCare':
        return None
    x1 = int(round(float(parts[4])))
    y1 = int(round(float(parts[5])))
    x2 = int(round(float(parts[6])))
    y2 = int(round(float(parts[7])))
    return cls, [x1, y1, x2, y2]


def load_ground_truth(gt_dir):
    gt_dir = Path(gt_dir)
    gt_by_image = {}
    for txt in sorted(gt_dir.glob('*.txt')):
        image_stem = txt.stem  # e.g., 000010
        with open(txt, 'r') as f:
            boxes = []
            for line in f:
                parsed = parse_kitti_label_line(line)
                if parsed is None:
                    continue
                cls, bbox = parsed
                boxes.append({'cls': cls, 'bbox': bbox, 'used': False})
        gt_by_image[image_stem] = boxes
    return gt_by_image


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_2d(pred_per_image, gt_by_image, classes, iou_thresh, img_ext):
    npos = {c: 0 for c in classes}
    for gts in gt_by_image.values():
        for g in gts:
            if g['cls'] in classes:
                npos[g['cls']] += 1

    ap_per_class = {}
    prec_per_class = {}
    rec_per_class = {}

    for cls in classes:
        records = []
        for image_name, plist in pred_per_image.items():
            stem = Path(image_name).stem
            for p in plist:
                if p['cls'] == cls:
                    records.append((stem, p['score'], p['bbox']))

        if not records:
            ap_per_class[cls] = 0.0
            prec_per_class[cls] = np.array([0.0])
            rec_per_class[cls] = np.array([0.0])
            continue

        records.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(records))
        fp = np.zeros(len(records))

        for gts in gt_by_image.values():
            for g in gts:
                if g['cls'] == cls:
                    g['used'] = False

        for i, (stem, score, pb) in enumerate(records):
            gts = [g for g in gt_by_image.get(stem, []) if g['cls'] == cls]
            ious = [iou_xyxy(pb, g['bbox']) for g in gts]
            if not ious:
                fp[i] = 1
                continue
            jmax = int(np.argmax(ious))
            iou_max = ious[jmax]
            if iou_max >= iou_thresh and not gts[jmax]['used']:
                tp[i] = 1
                gts[jmax]['used'] = True
            else:
                fp[i] = 1

        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        recall = tp_cum / max(npos[cls], 1e-12)
        ap = compute_ap(recall, precision)

        ap_per_class[cls] = ap
        prec_per_class[cls] = precision
        rec_per_class[cls] = recall

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return ap_per_class, mAP, prec_per_class, rec_per_class, npos


def main():
    args = parse_args()

    pred_per_image, pred_classes = load_predictions(args.pred_json)
    gt_by_image = load_ground_truth(args.gt_dir)

    classes = args.classes if args.classes else pred_classes

    print('\n--- Data Stats ---')
    counts_pred = {cls: 0 for cls in classes}
    for plist in pred_per_image.values():
        for p in plist:
            if p['cls'] in classes:
                counts_pred[p['cls']] += 1
    counts_gt = {cls: 0 for cls in classes}
    for gts in gt_by_image.values():
        for g in gts:
            if g['cls'] in classes:
                counts_gt[g['cls']] += 1
    for cls in classes:
        print(f"Class '{cls}': Preds={counts_pred[cls]}, GT={counts_gt[cls]}")
    print('---')

    ap_per_class, mAP, prec_per_class, rec_per_class, npos = evaluate_2d(
        pred_per_image=pred_per_image,
        gt_by_image=gt_by_image,
        classes=classes,
        iou_thresh=args.iou_thresh,
        img_ext=args.img_ext
    )

    print('Evaluation (IoU >= {:.2f})'.format(args.iou_thresh))
    for cls in classes:
        ap = ap_per_class.get(cls, 0.0)
        total = npos.get(cls, 0)
        last_p = float(prec_per_class[cls][-1]) if len(prec_per_class[cls]) else 0.0
        last_r = float(rec_per_class[cls][-1]) if len(rec_per_class[cls]) else 0.0
        f1 = 2 * last_p * last_r / (last_p + last_r) if (last_p + last_r) > 0 else 0.0
        print(f'- {cls}: AP={ap:.4f}, Recall={last_r:.4f}, Precision={last_p:.4f}, F1={f1:.4f}, GT={total}')
    print(f'mAP: {mAP:.4f}')


if __name__ == '__main__':
    main()
