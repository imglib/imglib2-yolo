import numpy as np
from typing import TYPE_CHECKING


###############################################################################
# HELPERS
###############################################################################

def filter_by_area(predictions: list, min_area: int) -> list:
	"""Remove detections whose bounding-box area is below min_area (pixels²)."""
	if min_area <= 0:
		return predictions
	def bbox_area(pred) -> float:
		return (b.x2 - b.x1) * (b.y2 - b.y1)
	return [p for p in predictions if bbox_area(p) >= min_area]


def predictions_to_table(predictions: list) -> list[dict]:
    """Serialise detections to a JSON-safe list of dicts."""
    rows = []
    boxes = predictions[0].boxes
    cls_names = predictions[0].names
    if boxes is not None:
        confs     = boxes.conf.cpu().tolist()
        xyxys     = boxes.xyxy.cpu().tolist()
        classes     = boxes.cls.cpu().tolist()
    ind = 0
    for conf, xyxy, cls in zip(confs, xyxys, classes):
        rows.append({
            "id":         ind,
            "class_id":  int(cls),
            "class_name": cls_names[int(cls)],
            "score":      conf,
            "x1":         xyxy[0],
            "y1":         xyxy[1],
            "x2":         xyxy[2],
            "y2":         xyxy[3],
        })
        ind = ind + 1 
    return rows


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
    conf: float   = globals()['conf']
    imgsz: int    = globals()['imgsz']
    min_area: int                   = globals()['min_area']
    use_gpu: bool                   = globals()['use_gpu']
else:
	import os
	sample_folder = '../../../samples/'
	#source_image  = io.imread(os.path.join(sample_folder, 'cycling001-1024x683.jpg'))
	conf          = 0.25
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

# Check image dimensions
msg_prefix = "YOLO: "
task.update(message=f"{msg_prefix}Input image shape {source_image.shape} with size {imgsz}")

# Only 1 plane -> ndims is 3, otherwise 4 [3, N, H, W]
n_planes = 1 if source_image.ndim == 3 else source_image.shape[1]
task.update( message=f"{msg_prefix}Input image has {n_planes} plane" + "s" if n_planes > 1 else "")

# Convert to [ N, H, W, 3 ] or [ H, W, 3 ] 
source_image = np.moveaxis(source_image, 0, -1)
task.update( message=f"{msg_prefix}Image shape after moveaxis: {source_image.shape}")

all_detections = model.predict( source_image, conf=conf, imgsz=imgsz, save=False )
all_detections = predictions_to_table( all_detections )
all_detections = filter_by_area(all_detections, min_area)

task.update( message = f"{msg_prefix}Done – {sum(len(p) for p in all_detections)} total detections")

# ── Return results ────────────────────────────────────────────────────────────
if appose_mode:
	task.outputs['detections'] = [all_detections]
else:
	import json
	print(json.dumps(all_detections, indent=2))
