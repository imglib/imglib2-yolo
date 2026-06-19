package net.imglib2.yolo;

public enum YOLOBuiltinModels
{
    // YOLO11 detection
    YOLO11N( "yolo11n.pt" ),
    YOLO11S( "yolo11s.pt" ),
    YOLO11M( "yolo11m.pt" ),
    YOLO11L( "yolo11l.pt" ),
    YOLO11X( "yolo11x.pt" ),

    // YOLOv8 detection
    YOLOV8N( "yolov8n.pt" ),
    YOLOV8S( "yolov8s.pt" ),
    YOLOV8M( "yolov8m.pt" ),
    YOLOV8L( "yolov8l.pt" ),
	YOLOV8X( "yolov8x.pt" ),
	
	YOLO26N("yolo26n.pt");

    private final String modelFile;

    YOLOBuiltinModels( final String modelFile )
    {
        this.modelFile = modelFile;
    }

    /** Model filename; downloaded automatically by Ultralytics on first use. */
    public String modelFile()
    {
        return modelFile;
    }
}