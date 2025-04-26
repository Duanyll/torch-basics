from pydantic import BaseModel

class CollageConfig(BaseModel):
    # Affine
    min_object_area: float = 0.001
    max_drop_area: float = 0.2
    chance_keep_leaf: float = 0.9
    chance_keep_stem: float = 0.2
    chance_split_stem: float = 0.75
    num_estimate_affine_samples: int = 50
    transform_erode_size: int = 3

    # Flow
    median_filter_kernel_size: int = 5
    low_motion_threshold_frame5: float = 0.007
    low_motion_threshold_final: float = 0.03
    stable_flow_threshold: float = 0.5

    # Palette
    delta_e_threshold: float = 15.0
    max_cluster_samples: int = 100000
    palette_spatial_weight: float = 0.5
    palette_area_threshold: float = 0.03
    palette_per_mask: int = 3
    num_palette_fallback: int = 5

    # Video
    min_frames: int = 5
    max_frames: int = 60
    chance_reverse: float = 0.5
    resolutions_720p: list[tuple[int, int]] = [
        # width, height
        (768, 768),
        (832, 704),
        (896, 640),
        (960, 576),
        (1024, 512),
    ]
    resolutions_1080p: list[tuple[int, int]] = [
        (1024, 1024),
        (1088, 960),
        (1152, 896),
        (1216, 832),
        (1280, 768),
        (1344, 704),
    ]
    chance_portrait: float = 0.2
    num_extract_attempts: int = 5
    
    # Pipeline
    chance_splat: float = 0.4
    chance_mask_edges: float = 0.4