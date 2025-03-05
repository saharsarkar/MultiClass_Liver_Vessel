# -*- coding: utf-8 -*-
"""
@author: S.Sarkar

"""

def multi_class_iou(y_true, y_pred, classes=[1, 2], threshold=0.5, smooth=1e-8):
    iou = []
    for class_id in classes:
        # Get ground truth and prediction for the specific class
        y_true_class = (y_true == class_id).float()
        y_pred_class = ((y_pred == class_id).float() > threshold).float()
        
        # Calculate intersection and union
        intersection = torch.sum(y_true_class * y_pred_class)
        union = torch.sum(y_true_class) + torch.sum(y_pred_class) - intersection
        
        # Calculate IoU with smoothing
        iou.append((intersection + smooth) / (union + smooth))
    return iou[0], iou[1]


def f1_score(y_true, y_pred, threshold=0.5, smooth=1e-8):
    y_pred_binary = (y_pred > threshold).float()
    
    true_positive = torch.sum(y_true * y_pred_binary)
    false_positive = torch.sum((1 - y_true) * y_pred_binary)
    false_negative = torch.sum(y_true * (1 - y_pred_binary))

    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1

def calculate_sensitivity(y_pred_class, arg_y, classes=[1, 2]):
    sensitivities = []
    for class_id in classes:
        # Get true positives and false negatives for the specific class
        TP = torch.sum((y_pred_class == class_id) & (arg_y == class_id))
        FN = torch.sum((y_pred_class != class_id) & (arg_y == class_id))
        
        # Calculate sensitivity with smoothing to avoid division by zero
        sensitivity = TP / (TP + FN + 1e-8)
        sensitivities.append(sensitivity)
    return sensitivities[0], sensitivities[1]


def calculate_specificity(y_pred_class, arg_y, classes=[1, 2]):
    specificities = []
    for class_id in classes:
        # Get true negatives and false positives for the specific class
        TN = torch.sum((y_pred_class != class_id) & (arg_y != class_id))
        FP = torch.sum((y_pred_class == class_id) & (arg_y != class_id))
        
        # Calculate specificity with smoothing to avoid division by zero
        specificity = TN / (TN + FP + 1e-8)
        specificities.append(specificity)
    return specificities


def dice_coef(y_true, y_pred, important_classes=[1, 2], smooth=1):
    dice_scores = []

    for class_id in important_classes:
        y_true_class = (y_true == class_id).float()
        y_pred_class = (y_pred == class_id).float()

        y_true_f = y_true_class.view(-1)
        y_pred_f = y_pred_class.view(-1)

        intersection = torch.sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

        dice_scores.append(dice_score)
    return dice_scores[0], dice_scores[1]
    #return torch.tensor(dice_scores)  # Convert the list to a PyTorch tensor