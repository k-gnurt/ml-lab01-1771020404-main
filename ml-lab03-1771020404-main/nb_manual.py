"""Các hàm nhỏ giúp hiểu cơ chế Gaussian Naive Bayes.

Sinh viên chỉ hoàn thiện hai hàm dưới đây; không cần tự cài toàn bộ GaussianNB.
"""

import math
import numpy as np


def compute_class_priors(y):
    """Trả về dict {class_label: prior_probability}."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    priors = {c: count / total_samples for c, count in zip(classes, counts)}
    return priors


def gaussian_log_likelihood(x, mean, var):
    """Tính log p(x | class) với giả định Gaussian một chiều.

    Công thức:
        -0.5 * log(2*pi*var) - (x-mean)^2 / (2*var)
    """
    if var <= 0:
        raise ValueError("Phương sai (var) phải lớn hơn 0.")
    
    log_likelihood = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
    return log_likelihood