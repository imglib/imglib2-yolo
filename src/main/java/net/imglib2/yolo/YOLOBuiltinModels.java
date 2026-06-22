package net.imglib2.yolo;

public enum YOLOBuiltinModels
{

	// YOLO26 detection
	YOLO26N( "yolo26n.pt" ),
	YOLO26S( "yolo26s.pt" ),
	YOLO26M( "yolo26m.pt" ),
	YOLO26L( "yolo26l.pt" ),
	YOLO26X( "yolo26x.pt" ),

	// YOLO11 detection
	YOLO11N( "yolo11n.pt" ),
	YOLO11S( "yolo11s.pt" ),
	YOLO11M( "yolo11m.pt" ),
	YOLO11L( "yolo11l.pt" ),
	YOLO11X( "yolo11x.pt" ),
	;

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
