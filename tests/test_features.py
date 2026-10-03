
import pytest
import pandas as pd
from src.features import build_feature_table  # adjust import to your structure


def test_build_feature_table_row_count_matches_images():
    """Positive test: one output row per image, even with zero annotations."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}, {'id': 2, 'file_name': 'b.jpg'}],
        'annotations': [{'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]}],
        'categories': [{'id': 0, 'name': 'wrinkle'}],
    }
    result = build_feature_table(fake_coco, 'test')
    assert len(result) == 2


def test_build_feature_table_zero_annotations_gives_zero_severity():
    """Positive test: an image with no annotations scores severity 0, not NaN."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [],
        'categories': [],
    }
    result = build_feature_table(fake_coco, 'test')
    assert result.loc[0, 'severity_score'] == 0


def test_build_feature_table_rejects_missing_categories():
    """Negative test: an annotation referencing an undefined category_id
    should raise, not silently produce a wrong class count."""
    fake_coco = {
        'images': [{'id': 1, 'file_name': 'a.jpg'}],
        'annotations': [{'image_id': 1, 'category_id': 99, 'bbox': [0, 0, 1, 1]}],
        'categories': [{'id': 0, 'name': 'wrinkle'}],
    }
    with pytest.raises(KeyError):
        build_feature_table(fake_coco, 'test')
