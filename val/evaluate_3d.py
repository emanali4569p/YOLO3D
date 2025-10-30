import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D detections (BEV/3D AP & mAP)')
    parser.add_argument('--pred_json', type=str, required=True, help='Path to predictions JSON (from inference.py)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to KITTI label_2 directory')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold (default 0.5)')
    parser.add_argument('--metric', type=str, default='bev', choices=['bev', '3d'], help='Metric: bev or 3d IoU')
    parser.add_argument('--classes', type=str, nargs='*', default=None, help='Optional class names to evaluate')
    return parser.parse_args()


# ======== Load Predictions ========
def load_predictions(pred_json):
    with open(pred_json, 'r') as f:
        preds = json.load(f)
    per_image = defaultdict(list)
    class_names = set()
    for p in preds:
        image = Path(p['image']).stem
        cls_name = str(p.get('class_name', '')).strip().capitalize()  # normalize to match KITTI labels
        score = float(p.get('score', 1.0))
        dims = p.get('dimensions')
        loc = p.get('location')
        ry = p.get('rotation_y')

        if dims is None or loc is None or ry is None:
            continue

        per_image[image].append({
            'cls': cls_name,
            'score': score,
            'hwl': np.array(dims, dtype=np.float32),
            'loc': np.array(loc, dtype=np.float32),
            'ry': float(ry)
        })
        class_names.add(cls_name)
    return per_image, sorted(class_names)


# ======== Load Ground Truth ========
def parse_kitti_label_line(line):
    parts = line.strip().split(' ')
    if len(parts) < 15:
        return None
    cls = parts[0].strip()
    if cls == 'DontCare':
        return None
    h = float(parts[8]); w = float(parts[9]); l = float(parts[10])
    x = float(parts[11]); y = float(parts[12]); z = float(parts[13])
    ry = float(parts[14])
    return cls, np.array([h, w, l], dtype=np.float32), np.array([x, y, z], dtype=np.float32), float(ry)


def load_ground_truth(gt_dir):
    gt_dir = Path(gt_dir)
    gt_by_image = {}
    all_classes = set()
    for txt in sorted(gt_dir.glob('*.txt')):
        stem = txt.stem
        entries = []
        with open(txt, 'r') as f:
            for line in f:
                parsed = parse_kitti_label_line(line)
                if parsed is None:
                    continue
                cls, hwl, loc, ry = parsed
                entries.append({'cls': cls, 'hwl': hwl, 'loc': loc, 'ry': ry, 'used': False})
                all_classes.add(cls)
        gt_by_image[stem] = entries
    return gt_by_image, sorted(all_classes)


# ======== Geometry ========
def bev_corners(loc, hwl, ry):
    w, l = hwl[1], hwl[2]
    x, z = loc[0], loc[2]
    dx, dz = l / 2.0, w / 2.0
    corners = np.array([[dx, dz], [dx, -dz], [-dx, -dz], [-dx, dz]], dtype=np.float32)
    c, s = np.cos(ry), np.sin(ry)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ rot.T + np.array([x, z])
    return rotated


def polygon_area(poly):
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def intersect_polygons(subject, clip):
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def intersection(a1, a2, b1, b2):
        s10 = a2 - a1; s32 = b2 - b1
        denom = s10[0]*s32[1] - s32[0]*s10[1]
        if abs(denom) < 1e-9:
            return a2
        t = ((b1[0]-a1[0])*s32[1] - (b1[1]-a1[1])*s32[0]) / denom
        return a1 + t * s10
    output = subject.copy()
    for i in range(len(clip)):
        input_list = output
        output = []
        A, B = clip[i], clip[(i + 1) % len(clip)]
        if len(input_list) == 0:
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    output.append(intersection(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(intersection(S, E, A, B))
            S = E
        output = np.array(output, dtype=np.float32)
    return output


def iou_bev(pred, gt):
    pc = bev_corners(pred['loc'], pred['hwl'], pred['ry'])
    gc = bev_corners(gt['loc'], gt['hwl'], gt['ry'])
    inter_poly = intersect_polygons(pc, gc)
    if inter_poly is None or len(inter_poly) == 0:
        return 0.0
    inter_area = polygon_area(inter_poly)
    area_p = polygon_area(pc)
    area_g = polygon_area(gc)
    union = area_p + area_g - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def iou_3d(pred, gt):
    pc = bev_corners(pred['loc'], pred['hwl'], pred['ry'])
    gc = bev_corners(gt['loc'], gt['hwl'], gt['ry'])
    inter_poly = intersect_polygons(pc, gc)
    if inter_poly is None or len(inter_poly) == 0:
        return 0.0
    inter_area = polygon_area(inter_poly)
    hp, hg = pred['hwl'][0], gt['hwl'][0]
    yp_bot, yp_top = pred['loc'][1] - hp / 2, pred['loc'][1] + hp / 2
    yg_bot, yg_top = gt['loc'][1] - hg / 2, gt['loc'][1] + hg / 2
    inter_h = max(0.0, min(yp_top, yg_top) - max(yp_bot, yg_bot))
    inter_vol = inter_area * inter_h
    vol_p = polygon_area(pc) * hp
    vol_g = polygon_area(gc) * hg
    union = vol_p + vol_g - inter_vol
    return float(inter_vol / union) if union > 0 else 0.0


# ======== Evaluation ========
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def evaluate(pred_per_image, gt_by_image, classes, iou_thresh, metric):
    npos = {c: 0 for c in classes}
    for gts in gt_by_image.values():
        for g in gts:
            if g['cls'] in classes:
                npos[g['cls']] += 1

    ap_per_class, prec_per_class, rec_per_class = {}, {}, {}
    iou_fn = iou_bev if metric == 'bev' else iou_3d

    for cls in classes:
        records = []
        for image, plist in pred_per_image.items():
            for p in plist:
                if p['cls'] == cls:
                    records.append((image, p['score'], p))
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

        for i, (stem, score, pobj) in enumerate(records):
            gts = [g for g in gt_by_image.get(stem, []) if g['cls'] == cls]
            if not gts:
                fp[i] = 1
                continue
            ious = [iou_fn(pobj, g) for g in gts]
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


# ======== Main ========
def main():
    args = parse_args()
    pred_per_image, pred_classes = load_predictions(args.pred_json)
    gt_by_image, gt_classes = load_ground_truth(args.gt_dir)

    classes = args.classes if args.classes else gt_classes

    print('\n--- Data Stats ---')
    for cls in classes:
        pred_count = sum(p['cls'] == cls for plist in pred_per_image.values() for p in plist)
        gt_count = sum(g['cls'] == cls for gts in gt_by_image.values() for g in gts)
        print(f"Class '{cls}': Preds={pred_count}, GT={gt_count}")
    print('---')

    ap_per_class, mAP, prec_per_class, rec_per_class, npos = evaluate(
        pred_per_image, gt_by_image, classes, args.iou_thresh, args.metric
    )

    print(f'\n3D Evaluation ({args.metric.upper()} IoU >= {args.iou_thresh:.2f})')
    for cls in classes:
        ap = ap_per_class.get(cls, 0.0)
        r = rec_per_class[cls][-1] if len(rec_per_class[cls]) else 0
        p = prec_per_class[cls][-1] if len(prec_per_class[cls]) else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        print(f"- {cls}: AP={ap:.4f}, Recall={r:.4f}, Precision={p:.4f}, F1={f1:.4f}, GT={npos[cls]}")
    print(f"\nMean AP (mAP): {mAP:.4f}")


if __name__ == '__main__':
    main()
