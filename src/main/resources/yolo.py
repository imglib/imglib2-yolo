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
		b = pred.bbox
		return (b.maxx - b.minx) * (b.maxy - b.miny)
	return [p for p in predictions if bbox_area(p) >= min_area]


def predictions_to_table(predictions: list, plane_index: int = 0) -> list[dict]:
	"""Serialise detections to a JSON-safe list of dicts."""
	rows = []
	for i, pred in enumerate(predictions, start=1):
		b = pred.bbox
		rows.append({
			"id":         i,
			"class_id":   int(pred.category.id),
			"class_name": pred.category.name,
			"score":      pred.score.value,
			"x1":         b.minx,
			"y1":         b.miny,
			"x2":         b.maxx,
			"y2":         b.maxy,
		})
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
task.update(message=f"{msg_prefix}Input image shape {source_image.shape}")


kwargs = dict(
	conf  = conf,
    imgsz = imgsz,
)

# Only 1 plane -> ndims is 3, otherwise 4 [3, N, H, W]
n_planes = 1 if source_image.ndim == 3 else source_image.shape[1]
task.update( message=f"{msg_prefix}Input image has {n_planes} plane" + "s" if n_planes > 1 else "")

# Convert to [ N, H, W, 3 ] or [ H, W, 3 ] for SAHI
source_image = np.moveaxis(source_image, 0, -1)
task.update( message=f"{msg_prefix}Image shape after moveaxis: {source_image.shape}")

all_detections = model.predict( source_image, kwargs )
all_detections = filter_by_area(all_detections, min_area)

task.update( message = f"{msg_prefix}Done – {sum(len(p) for p in all_detections)} total detections")

# ── Return results ────────────────────────────────────────────────────────────
if appose_mode:
	task.outputs['detections'] = all_detections
else:
	import json
	print(json.dumps(all_detections, indent=2))
