import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

resize0 = A.Compose([
    A.Resize(height=224, width=224),
    A.Rotate(limit=(-45, 45), p=1, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.Normalize(normalization="min_max", p=1.0),
    ToTensorV2(),
])
