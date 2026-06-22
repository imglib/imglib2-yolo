import numpy as np
from typing import TYPE_CHECKING
from sahi.predict import get_sliced_prediction, get_prediction


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
# INFERENCE
###############################################################################

def run_yolo_sahi(stack: np.ndarray, model, kwargs: dict) -> list:

	# Override confidence threshold for this run
	model.confidence_threshold = kwargs['conf']
	use_sahi = kwargs['use_sahi']
	msg_prefix = "YOLO-SAHI: "
	results = []
	n_planes = 1 if stack.ndim == 3 else stack.shape[0]

	if use_sahi:
		task.update(message=f"{msg_prefix}Running YOLO-SAHI detection on image of shape {stack.shape}")
		for i in range(n_planes):
			plane = stack[i] if n_planes > 1 else stack
			result = get_sliced_prediction(
				image                       = plane,
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
			results.append(predictions_to_table(result.object_prediction_list))
			task.update(current=i+1, maximum = n_planes)
	else:
		task.update(message=f"{msg_prefix}Running standard YOLO detection on image of shape {stack.shape}")
		for i in range(n_planes):
			plane = stack[i] if n_planes > 1 else stack
			result = get_prediction(
				image=plane,
				detection_model=model,
				verbose=0,
			)
			results.append(predictions_to_table(result.object_prediction_list))
			task.update(current=i+1, maximum = n_planes)
	
	return results


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
	source_image  = io.imread(os.path.join(sample_folder, 'cycling001-1024x683.jpg'))
	conf          = 0.25
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

# Check image dimensions
msg_prefix = "YOLO-SAHI: "
task.update(message=f"{msg_prefix}Input image shape {source_image.shape}")

kwargs = dict(
	conf                        = conf,
	use_sahi                    = use_sahi,
	slice_height                = slice_height,
	slice_width                 = slice_width,
	overlap_height_ratio        = overlap_height_ratio,
	overlap_width_ratio         = overlap_width_ratio,
	perform_standard_pred       = perform_standard_pred,
	postprocess_type            = postprocess_type,
	postprocess_match_threshold = postprocess_match_threshold,
)

# Only 1 plane -> ndims is 3, otherwise 4 [3, N, H, W]
n_planes = 1 if source_image.ndim == 3 else source_image.shape[1]
task.update( message=f"{msg_prefix}Input image has {n_planes} plane" + "s" if n_planes > 1 else "")

# Convert to [ N, H, W, 3 ] or [ H, W, 3 ] for SAHI
source_image = np.moveaxis(source_image, 0, -1)
task.update( message=f"{msg_prefix}Image shape after moveaxis: {source_image.shape}")

all_detections = run_yolo_sahi(source_image, model, kwargs)
all_detections = filter_by_area(all_detections, min_area)

task.update( message = f"{msg_prefix}Done – {sum(len(p) for p in all_detections)} total detections")

# ── Return results ────────────────────────────────────────────────────────────
if appose_mode:
	task.outputs['detections'] = all_detections
else:
	import json
	print(json.dumps(all_detections, indent=2))
