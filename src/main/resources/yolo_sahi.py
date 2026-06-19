###
# Runs YOLO detection with optional SAHI slicing.
# Detection only: outputs a list of {id, class_name, score, x1, y1, x2, y2}
# returned via task.export(), no shared-memory label image needed.
###
import numpy as np
from typing import TYPE_CHECKING
from sahi.predict import get_sliced_prediction, get_prediction


###############################################################################
# HELPERS
###############################################################################

def assemble_rgb(
        image: np.ndarray,
        chan_r: int,
        chan_g: int,
        chan_b: int ) -> np.ndarray:
    """
    Build a (H, W, 3) uint8 RGB image from a 3-channel source array.
    Input is expected to be [3, H, W] (channels first, as from ImgLib2 Views.stack).
    chan_r/g/b are 1-based indices into the channel axis (axis 0); 0 = fill plane with zeros.
    Each plane is independently normalised to [0, 255].
    """
    h, w = image.shape[1:]

    def pick(idx: int) -> np.ndarray:
        if idx == 0:
            return np.zeros((h, w), dtype=np.float32)
        plane = image[idx - 1, :, :].astype(np.float32)
        pmin = plane.min()
        pmax = plane.max()
        ptp = pmax - pmin
        return (plane - pmin) / (ptp + 1e-8) * 255

    return np.stack([pick(chan_r), pick(chan_g), pick(chan_b)], axis=-1).astype(np.uint8)


def filter_by_area(predictions: list, min_area: int) -> list:
    """Remove detections whose bounding-box area is below min_area (pixels²)."""
    if min_area <= 0:
        return predictions
    def bbox_area(pred) -> float:
        b = pred.bbox
        return (b.maxx - b.minx) * (b.maxy - b.miny)
    return [p for p in predictions if bbox_area(p) >= min_area]


def predictions_to_table(predictions: list, plane_index: int = 0) -> list[dict]:
    """Serialise detections to a JSON-safe list of dicts."""
    rows = []
    for i, pred in enumerate(predictions, start=1):
        b = pred.bbox
        rows.append({
            "plane":      plane_index,
            "id":         i,
            "class_id":   pred.category.id,
            "class_name": pred.category.name,
            "score":      float(pred.score.value),
            "x1": float(b.minx), "y1": float(b.miny),
            "x2": float(b.maxx), "y2": float(b.maxy),
        })
    return rows


###############################################################################
# INFERENCE
###############################################################################

def run_yolo_sahi(rgb_image: np.ndarray, model, kwargs: dict) -> list:
    """Run YOLO detection on a single (H, W, 3) uint8 image."""

    # Override confidence threshold for this run
    model.confidence_threshold = kwargs['conf']

    if kwargs['use_sahi']:
        result = get_sliced_prediction(
            image                       = rgb_image,
            detection_model             = model,
            slice_height                = kwargs['slice_height'],
            slice_width                 = kwargs['slice_width'],
            overlap_height_ratio        = kwargs['overlap_height_ratio'],
            overlap_width_ratio         = kwargs['overlap_width_ratio'],
            perform_standard_pred       = kwargs['perform_standard_pred'],
            postprocess_type            = kwargs['postprocess_type'],
            postprocess_match_threshold = kwargs['postprocess_match_threshold'],
            postprocess_match_metric    = 'IOU',
            verbose                     = 0,
        )
    else:
        result = get_prediction(
            image           = rgb_image,
            detection_model = model,
            verbose         = 0,
        )

    return result.object_prediction_list


###############################################################################
# MAIN
###############################################################################

appose_mode = 'task' in globals()
if appose_mode:
    if TYPE_CHECKING:
        from appose.python_worker import Task
        task: Task
    task = globals()['task']
else:
    from appose.python_worker import Task
    from yolo_utils import get_torch_device
    import os
    task = Task()

# ── Load parameters ───────────────────────────────────────────────────────────
if appose_mode:
    source_image  = globals()['input'].ndarray()
    chan_r: int   = globals()['chan_r']
    chan_g: int   = globals()['chan_g']
    chan_b: int   = globals()['chan_b']
    conf: float   = globals()['conf']
    iou: float    = globals()['iou']
    imgsz: int    = globals()['imgsz']
    use_sahi: bool                  = globals()['use_sahi']
    slice_height: int               = globals()['slice_height']
    slice_width: int                = globals()['slice_width']
    overlap_height_ratio: float     = globals()['overlap_height_ratio']
    overlap_width_ratio: float      = globals()['overlap_width_ratio']
    perform_standard_pred: bool     = globals()['perform_standard_pred']
    postprocess_type: str           = globals()['postprocess_type']
    postprocess_match_threshold: float = globals()['postprocess_match_threshold']
    min_area: int                   = globals()['min_area']
    use_gpu: bool                   = globals()['use_gpu']
else:
    from sahi.utils.cv import read_image
    import os
    sample_folder = '../../../samples/'
    source_image  = io.imread(os.path.join(sample_folder, 'test_XY.tif'))
    chan_r, chan_g, chan_b = 1, 2, 0
    conf          = 0.25
    iou           = 0.7
    imgsz         = 640
    use_sahi      = True
    slice_height  = 640
    slice_width   = 640
    overlap_height_ratio  = 0.2
    overlap_width_ratio   = 0.2
    perform_standard_pred = True
    postprocess_type      = 'NMW'
    postprocess_match_threshold = 0.5
    min_area      = 0
    use_gpu       = False

# ── Device ────────────────────────────────────────────────────────────────────
use_gpu, device = get_torch_device(use_gpu)

# ── Retrieve pre-loaded model ─────────────────────────────────────────────────
model = globals().get('model', None)
if model is None:
    raise RuntimeError(
        "YOLO model not found. "
        "Make sure yolo_sahi_init.py ran first and exported 'model'."
    )

kwargs = dict(
    conf                        = conf,
    iou                         = iou,
    imgsz                       = imgsz,
    use_sahi                    = use_sahi,
    slice_height                = slice_height,
    slice_width                 = slice_width,
    overlap_height_ratio        = overlap_height_ratio,
    overlap_width_ratio         = overlap_width_ratio,
    perform_standard_pred       = perform_standard_pred,
    postprocess_type            = postprocess_type,
    postprocess_match_threshold = postprocess_match_threshold,
)

# ── Loop over all planes (YOLO is always 2D) ─────────────────────────────────
# Input layout from ImgLib2 argbToRGBStack: [3, X, Y] for 2D, or [3, X, Y, Z, ...] for higher-D
# The channel axis is always first (axis 0). All remaining axes are spatial.
# We iterate over the last two spatial axes (Y, X are always last), treating any
# axes beyond the channel and spatial dimensions as planes to iterate over.

# Determine the number of planes by looking at dimensions beyond [3, H, W]
# For [3, H, W]: single plane
# For [3, X, Y, Z]: Z planes, each is [3, X, Y]
# For [3, X, Y, Z, T]: Z*T planes, etc.

# Simplified: treat everything after the first 3 dimensions as planes to iterate
# [3, H, W] -> 1 plane
# [3, H, W, D] -> D planes, each [3, H, W]
# [3, H, W, D, E] -> D*E planes, each [3, H, W]

if source_image.ndim == 3:
    # Single RGB image: [3, H, W]
    planes = [source_image]
    n_planes = 1
else:
    # Multi-dimensional stack: [3, H, W, ...]
    # Reshape to [N, 3, H, W] where N = product of all extra dimensions
    h, w = source_image.shape[1], source_image.shape[2]
    extra_dims = source_image.shape[3:]
    n_planes = 1
    for d in extra_dims:
        n_planes *= d

    # Reshape to [N, 3, H, W]
    reshaped = source_image.reshape((n_planes, 3, h, w))
    planes = [reshaped[i] for i in range(n_planes)]

task.update(current=0, maximum=n_planes,
            message=f"YOLO+SAHI: Starting – {n_planes} plane(s), device={device}")

all_detections = []

for plane_idx, plane in enumerate(planes):
    rgb = assemble_rgb(plane, chan_r, chan_g, chan_b)

    predictions = run_yolo_sahi(rgb, model, kwargs)
    predictions = filter_by_area(predictions, min_area)

    all_detections.extend(predictions_to_table(predictions, plane_index=plane_idx))

    task.update(
        current = plane_idx + 1,
        maximum = n_planes,
        message = f"YOLO+SAHI: Plane {plane_idx + 1}/{n_planes} – {len(predictions)} detections"
    )

task.update(
    current = n_planes,
    maximum = n_planes,
    message = f"YOLO+SAHI: Done – {len(all_detections)} total detections"
)

# ── Return results ────────────────────────────────────────────────────────────
if appose_mode:
    task.update(message=f"Exporting {len(all_detections)} detections")
    task.export( detections=all_detections )
else:
    import json
    print(json.dumps(all_detections, indent=2))
