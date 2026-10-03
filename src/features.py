from collections import defaultdict
import pandas as pd


def build_feature_table(coco_data, split_name, categories):
    """Convert one split's COCO annotation data into a per-image feature table.

    EXPERIMENTAL VERSION (branch: feature/alt-severity-score): severity_score
    is computed as the TOTAL bounding-box area across all detections in an
    image, rather than the mean. This weights images with many small defects
    more heavily than the mean-based version does, since it doesn't average
    out multiple detections -- testing whether this changes model behaviour
    or performance compared to the mean-based definition used on main.

    Parameters
    ----------
    coco_data : dict
        Parsed COCO-format annotation data for one split.
    split_name : str
        Label recorded in the output table's 'split' column.
    categories : dict
        Mapping of category_id -> class name.

    Returns
    -------
    pandas.DataFrame
        One row per image, with per-class defect counts as input features
        and a continuous 'severity_score' target column.
    """
    image_info = {img['id']: img['file_name'] for img in coco_data['images']}

    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    rows = []
    for image_id, file_name in image_info.items():
        anns = anns_by_image.get(image_id, [])
        class_names = [categories[a['category_id']] for a in anns]

        row = {
            'image': file_name,
            'split': split_name,
            'num_detections': len(anns),
            'loose_meat_count': class_names.count('loose-meat'),
            'packaging_error_count': class_names.count('packaging-error'),
            'twisted_meat_count': class_names.count('twisted-meat'),
            'unsealed_count': class_names.count('unsealed'),
            'wrinkle_count': class_names.count('wrinkle'),
            # CHANGED: total area, not mean area (compare against main)
            'severity_score': sum(a['bbox'][2] * a['bbox'][3] for a in anns),
        }
        rows.append(row)

    return pd.DataFrame(rows)
