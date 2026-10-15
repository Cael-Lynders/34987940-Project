
import pytest
import pandas as pd
from src.features import build_feature_table


def test_build_feature_table_row_count_matches_images():
    """Positive test: one output row per image, even with zero annotations."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}, {'id': 2, 'file_name': 'b.jpg'}],
        'annotations': [{'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]}],
    }
    categories = {0: 'wrinkle'}
    result = build_feature_table(fake_coco, 'test', categories)
    assert len(result) == 2


def test_build_feature_table_zero_annotations_gives_zero_severity():
    """Positive test: an image with no annotations scores severity 0, not NaN."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [],
    }
    categories = {}
    result = build_feature_table(fake_coco, 'test', categories)
    assert result.loc[0, 'severity_score'] == 0


def test_build_feature_table_rejects_missing_categories():
    """Negative test: an annotation referencing an undefined category_id
    should raise, not silently produce a wrong class count."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [{'image_id': 1, 'category_id': 99, 'bbox': [0, 0, 1, 1]}],
    }
    categories = {0: 'wrinkle'}  # category 99 deliberately missing
    with pytest.raises(KeyError):
        build_feature_table(fake_coco, 'test', categories)

def test_severity_is_mean_of_box_areas():
    """Known-answer test: severity = mean(width * height) of an image's boxes."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]},
            {'image_id': 1, 'category_id': 0, 'bbox': [5, 5, 20, 20]},
        ],
    }
    result = build_feature_table(fake_coco, 'test', {0: 'wrinkle'})
    assert result.loc[0, 'severity_score'] == 250   # (100 + 400) / 2


def test_class_counts_match_annotations():
    """Known-answer test: per-class counts and total detections are exact."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 1, 1]},
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 1, 1]},
            {'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 1, 1]},
        ],
    }
    result = build_feature_table(fake_coco, 'test', {0: 'wrinkle', 1: 'unsealed'})
    assert result.loc[0, 'wrinkle_count'] == 2
    assert result.loc[0, 'unsealed_count'] == 1
    assert result.loc[0, 'num_detections'] == 3
