###
# Initialises the YOLO detection model 
# Run once; the model is exported and reused across calls.
###
from typing import TYPE_CHECKING
from ultralytics import YOLO

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
    model_file = 'yolo26n.pt'

if appose_mode:
    model_file: str = globals()['model_file']

task.update(message = f"YOLO: Loading model '{model_file}'")

model = YOLO().load( model_file )

task.update(message = "YOLO: Model ready")

if appose_mode:
    task.export( model=model )
