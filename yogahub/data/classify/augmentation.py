import albumentations as A
from albumentations.pytorch import ToTensorV2

# Declare an augmentation pipeline
train_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=420),
        # A.RandomCrop(420, 420),
        A.Resize(448, 448),
        A.RandomCrop(420, 420),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.HueSaturationValue(),
        A.OneOf(
            [
                A.GaussNoise(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.PiecewiseAffine(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.HueSaturationValue(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=420),
        # A.RandomCrop(420, 420),
        A.Resize(448, 448),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

visualize_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=420),
        # A.RandomCrop(420, 420),
        A.Resize(448, 448),
        A.RandomCrop(420, 420),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.HueSaturationValue(),
        A.OneOf(
            [
                A.GaussNoise(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.PiecewiseAffine(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.HueSaturationValue(),
    ]
)
