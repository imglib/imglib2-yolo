###
# Initialises the YOLO detection model via SAHI AutoDetectionModel.
# Run once; the model is exported and reused across calls.
###
from typing import TYPE_CHECKING
from sahi import AutoDetectionModel

appose_mode = 'task' in globals()
if appose_mode:
    if TYPE_CHECKING:
        from appose.python_worker import Task
        task: Task
    task = globals()['task']
else:
    from appose.python_worker import Task
    from yolo_utils import get_torch_device
    task = Task()
    model_file = 'yolo11m.pt'
    use_gpu    = False

if appose_mode:
    model_file: str = globals()['model_file']
    use_gpu: bool   = globals()['use_gpu']

use_gpu, device = get_torch_device(use_gpu)

task.update(message = f"YOLO-SAHI: Loading model '{model_file}' on {device}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type           = 'ultralytics',
    model_path           = model_file,
    confidence_threshold = 0.25,   # overridden per-run
    device               = str(device),
)

task.update(message = "YOLO-SAHI: Model ready")

if appose_mode:
    task.export( model=detection_model )
