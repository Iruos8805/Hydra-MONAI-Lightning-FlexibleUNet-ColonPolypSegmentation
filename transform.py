from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandZoomd,
    RandShiftIntensityd,
    RandAffined,
    EnsureTyped,
    Compose,
    Lambdad,
)

train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="label", func=lambda x: x[:1]),  # Force single channel
        Resized(keys=["image", "label"], spatial_size=(320, 320)),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        # Ensuring masks are also normalized to [0,1]
        ScaleIntensityRanged(
            keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.2,
            mode=("bilinear", "nearest"),
        ),
        RandAffined(
            keys=["image", "label"],
            rotate_range=[0.0, 0.0, 0.1],
            scale_range=[0.05, 0.05, 0.0],
            prob=0.2,
            mode=("bilinear", "nearest"),
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="label", func=lambda x: x[:1]),  # Force single channel
        Resized(keys=["image", "label"], spatial_size=(320, 320)),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        # Ensuring masks are also normalized to [0,1]
        ScaleIntensityRanged(
            keys=["label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)
