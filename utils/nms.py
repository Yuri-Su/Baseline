# -*- coding = utf-8 -*-
# @Time : 2021/12/14 7:59 PM
# @Author : Yuri Su
import tensorflow as tf
import numpy as np


def compute_iou(boxes, scores, iou, max_output_size):
    selected_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=iou,
                                                    max_output_size=max_output_size)
    return tf.gather(boxes, selected_indices)


if __name__ == '__main__':
    boxes = np.array([[1, 2, 3, 4], [1, 3, 3, 4], [1, 3, 4, 4], [1, 1, 4, 4], [1, 1, 4, 4]], dtype=np.float32)
    scores = np.array([0.4, 0.5, 0.72, 0.90, 45], dtype=np.float32)
    print(compute_iou(boxes, scores, 0.5, 5))
