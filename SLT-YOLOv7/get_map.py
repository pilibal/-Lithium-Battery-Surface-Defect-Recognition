import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import glob

from utils.utils import get_classes
from utils.utils_map1 import get_map
from yolo import YOLO


class PerImageMetricsCollector:
    """
    收集每个样本的详细指标，包含真实AP计算（PR曲线下面积）
    """

    def __init__(self, class_names, iou_threshold=0.5):
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.results = []

    def compute_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area
        return inter_area / (union_area + 1e-6)

    def calculate_ap_per_image(self, pred_boxes, gt_boxes, conf_thresholds=None):
        """
        计算单张图像的AP（Average Precision）
        通过在不同置信度阈值下计算PR曲线，然后求曲线下面积
        """
        if not pred_boxes or not gt_boxes:
            return 0.0

        if conf_thresholds is None:
            # 使用预测框的置信度作为阈值点
            confs = sorted(list(set([p['confidence'] for p in pred_boxes])), reverse=True)
            if len(confs) == 0:
                return 0.0
        else:
            confs = conf_thresholds

        precisions = []
        recalls = []

        for conf_thresh in confs:
            # 筛选高于阈值的预测
            filtered_preds = [p for p in pred_boxes if p['confidence'] >= conf_thresh]

            if not filtered_preds:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            # 计算TP、FP
            tp = 0
            fp = 0
            gt_matched = [False] * len(gt_boxes)

            # 按置信度排序
            filtered_preds.sort(key=lambda x: x['confidence'], reverse=True)

            for pred in filtered_preds:
                best_iou = 0
                best_gt_idx = -1

                for idx, gt in enumerate(gt_boxes):
                    if gt['class'] == pred['class'] and not gt_matched[idx]:
                        iou = self.compute_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_iou >= self.iou_threshold and best_gt_idx != -1:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        # 计算曲线下面积（使用VOC标准：插值法）
        if not precisions:
            return 0.0

        # 按recall排序
        sorted_pairs = sorted(zip(recalls, precisions))
        recalls = [r for r, p in sorted_pairs]
        precisions = [p for r, p in sorted_pairs]

        # 计算AP（使用11点插值或全点插值，这里使用全点插值）
        ap = 0.0
        prev_recall = 0.0

        for i in range(len(recalls)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i - 1])

        return ap

    def evaluate_image(self, image_id, gt_path, dr_path):
        """
        评估单张图片的指标（包含真实AP计算）
        """
        # 读取真实框
        gt_boxes = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_name = parts[0]
                        if 'difficult' in line:
                            continue
                        box = [float(x) for x in parts[1:5]]
                        gt_boxes.append({'class': cls_name, 'bbox': box, 'used': False})

        # 读取检测结果（保留所有置信度用于AP计算）
        pred_boxes = []
        if os.path.exists(dr_path):
            with open(dr_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls_name = parts[0]
                        conf = float(parts[1])
                        box = [float(x) for x in parts[2:6]]
                        pred_boxes.append({
                            'class': cls_name,
                            'bbox': box,
                            'confidence': conf
                        })

        # 计算基础指标（基于固定阈值0.5）
        tp_05 = 0
        fp_05 = 0
        pred_boxes_05 = [p for p in pred_boxes if p['confidence'] >= 0.5]

        gt_matched = [False] * len(gt_boxes)
        for pred in pred_boxes_05:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if gt['class'] == pred['class'] and not gt_matched[idx]:
                    iou = self.compute_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                tp_05 += 1
                gt_matched[best_gt_idx] = True
            else:
                fp_05 += 1

        fn = len(gt_boxes) - sum(gt_matched)

        precision = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0
        recall = tp_05 / len(gt_boxes) if len(gt_boxes) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 计算真实AP（基于所有置信度阈值）
        ap = self.calculate_ap_per_image(pred_boxes, gt_boxes)

        # 计算mAP近似（所有类别的平均AP，这里简化为单图多类别的平均）
        # 注意：单图的mAP通常意义不大，但可以作为该图检测质量的综合指标

        confidences = [p['confidence'] for p in pred_boxes]

        return {
            'image_id': image_id,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,  # 这是真实计算的AP，不是近似值
            'mAP_approx': ap,  # 单图级别，等同于AP
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes),
            'num_pred_05': len(pred_boxes_05),
            'num_tp': tp_05,
            'num_fp': fp_05,
            'num_fn': fn,
            'mean_confidence': np.mean(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0
        }

    def collect_metrics(self, map_out_path, image_ids):
        """收集所有图像的指标"""
        gt_dir = os.path.join(map_out_path, 'ground-truth')
        dr_dir = os.path.join(map_out_path, 'detection-results')

        metrics_list = []
        for image_id in tqdm(image_ids, desc="Computing per-image metrics (with AP)"):
            gt_file = os.path.join(gt_dir, f"{image_id}.txt")
            dr_file = os.path.join(dr_dir, f"{image_id}.txt")

            metrics = self.evaluate_image(image_id, gt_file, dr_file)
            metrics_list.append(metrics)

        self.results = metrics_list
        return metrics_list

    def save_to_csv(self, output_path):
        """保存为CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)

        # 打印摘要
        print(f"\nPer-image metrics saved to: {output_path}")
        print(f"Average metrics across {len(df)} images:")
        for col in ['precision', 'recall', 'f1', 'ap']:
            if col in df.columns:
                print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")

        return df


if __name__ == "__main__":
    # 配置参数
    map_mode = 0
    classes_path = 'model_data/voc_classes.txt'
    MINOVERLAP = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threhold = 0.5
    map_vis = False
    VOCdevkit_path = 'VOCdevkit'

    # 模型标识
    model_name = "epoch500"
    map_out_path = f'map_out/{model_name}_detailed1'

    # 创建目录
    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    if map_vis:
        os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)

    class_names, _ = get_classes(classes_path)

    # 读取图像列表
    test_txt = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")
    image_ids = open(test_txt).read().strip().split()

    # 1. 生成预测结果
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    # 2. 生成真实标签
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    # 3. 计算mAP和逐样本指标（包含真实AP）
    if map_mode == 0 or map_mode == 3:
        print("Get map.")

        # 3.1 计算整体mAP（类别级别）
        ap_dictionary = get_map(MINOVERLAP, True, score_threhold=score_threhold,
                                path=map_out_path, return_ap=True)

        # 3.2 计算逐样本指标（包含单图AP）
        print("\nComputing per-image metrics (including AP)...")
        collector = PerImageMetricsCollector(class_names, iou_threshold=MINOVERLAP)
        per_image_metrics = collector.collect_metrics(map_out_path, image_ids)

        # 保存详细CSV（现在包含ap列）
        csv_path = os.path.join(map_out_path, f"{model_name}_per_image_metrics.csv")
        df_metrics = collector.save_to_csv(csv_path)

        # 3.3 计算类别级mAP统计
        if ap_dictionary and isinstance(ap_dictionary, dict):
            ap_values = np.array(list(ap_dictionary.values()))
            ap_mean = np.mean(ap_values)  # 这就是mAP
            ap_std = np.std(ap_values)
            cv = ap_std / ap_mean if ap_mean > 0 else 0

            print("\n" + "=" * 60)
            print("类别级AP统计 (Class-level AP):")
            print(f"Classes: {len(ap_values)}")
            print(f"mAP: {ap_mean:.4f} ± {ap_std:.4f}")
            print(f"CV (变异系数): {cv:.4f}")
            print("=" * 60)

        # 3.4 图像级AP统计（新）
        if len(df_metrics) > 0 and 'ap' in df_metrics.columns:
            print("\n图像级AP统计 (Image-level AP):")
            print(f"Mean AP per image: {df_metrics['ap'].mean():.4f} ± {df_metrics['ap'].std():.4f}")
            print(f"AP Range: [{df_metrics['ap'].min():.4f}, {df_metrics['ap'].max():.4f}]")

        # 保存汇总
        summary_path = os.path.join(map_out_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'model': model_name,
                'class_level': {
                    'mAP': float(ap_mean) if 'ap_mean' in locals() else None,
                    'std': float(ap_std) if 'ap_std' in locals() else None,
                    'per_class_ap': {k: float(v) for k, v in ap_dictionary.items()} if ap_dictionary else {}
                },
                'image_level': {
                    'mean_ap': float(df_metrics['ap'].mean()),
                    'std_ap': float(df_metrics['ap'].std()),
                    'mean_f1': float(df_metrics['f1'].mean()),
                    'mean_precision': float(df_metrics['precision'].mean()),
                    'mean_recall': float(df_metrics['recall'].mean())
                }
            }, f, indent=2)

        print(f"\nAll results saved to: {map_out_path}")
        print(f"  - Per-image CSV: {csv_path}")
        print(f"  - Summary JSON: {summary_path}")
        print("Get map done.")