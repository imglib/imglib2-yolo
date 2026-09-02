package net.imglib2.yolo;

import java.util.HashMap;
import java.util.Map;

import net.imglib2.appose.ShmImg;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Interface for Yolo parameters common to all yolo versions */
public abstract class YOLOMainParameters 
{
	// ── Model ─────────────────────────────────────────────────────────────────
    public YOLOBuiltinModels builtinModel;

    public String customModel;
    
    public double scale; // scaling of the image to determine imgsz

    // ── Core inference ────────────────────────────────────────────────────────
    public double conf;


    // ── Advanced ──────────────────────────────────────────────────────────────
    public int minArea;

    // ── GPU ───────────────────────────────────────────────────────────────────
    public boolean useGpu;

    /** Default contrusctor */
    public YOLOMainParameters()
    {
    	this.builtinModel           = YOLOBuiltinModels.YOLO26N;
    	this.customModel                       = null;
    	this.scale 							   = 1.0;
    	this.conf                              = 0.1;
    	this.minArea                           = 0;
    	this.useGpu                           = true;
    }

    /** Constructor by copy  */
    public YOLOMainParameters( final YOLOMainParameters b )
    {
        this.builtinModel                = b.builtinModel;
        this.customModel                 = b.customModel;
        this.conf                        = b.conf;
        this.minArea                     = b.minArea;
        this.useGpu                      = b.useGpu;
        this.scale 						 = b.scale;
    }
    
    /** Constructor by values */
    public YOLOMainParameters( YOLOBuiltinModels builtinModel, String customModel, double scale, double conf, int minArea, boolean useGpu )
    {
        this.builtinModel                = builtinModel;
        this.customModel                 = customModel;
        this.conf                        = conf;
        this.minArea                     = minArea;
        this.useGpu                      = useGpu;
		this.scale 						 = scale;
    }
    
	/**  Calculates the imgsz parameter given the desired scaling.
	 * 
	 * Ideal imgsz should be such that: effective object size ≈ original object size × imgsz / input region size (image size) 
	 * 
	 * If scale = 1, assume object size is the same as the training objects size: no rescaling should be done so imgsz~imagesize
	 * In Yolo, the longest side is resized to imgsz
	 * If scale > 1, objects are bigger than training object size: input image should be downsized, imgsz < longest side
	 * If scale < 1, objects are smaller than training, input image should be made bigger, imgsz > longest side
	 * */
	public int calculate_imgsz( int img_width, int img_height )
	{
		int longest = img_width > img_height ? img_width: img_height;
		double target_size = longest / scale; 
		// imgsz should be a multiple of 32, so adjust the value to closest 32 integer
		target_size = target_size / 32.0;
        long div_size = Math.round(target_size);
        return (int) (div_size * 32);
	}
    
    /**
     * Builds the map passed to the Appose Python task.
     * Detections are returned via task.export() on the Python side,
     * so no output shared-memory image is needed.
     */
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input, final AxisInfo axisInfo )
    {
        final Map< String, Object > inputs = new HashMap<>();

        // ── Input image ───────────────────────────────────────────────────────
        inputs.put( "input", input.ndArray() );
        
        final int imgsz = calculate_imgsz( (int) input.dimension(axisInfo.X()), (int) input.dimension(axisInfo.Y()) ); // checker that dimensions are width and height 
		inputs.put( "imgsz", imgsz );
        
        // Axis position, if there are channels, time or Z
    	final AxisInfo axisInfoPython = axisInfo.toPython();
		inputs.put( "t_axis", axisInfoPython.T() < 0 ? null : axisInfoPython.T() );
		inputs.put( "z_axis", axisInfoPython.Z() < 0 ? null : axisInfoPython.Z() );
		inputs.put( "channel_axis", axisInfoPython.C() < 0 ? null : axisInfoPython.C() );

        // ── Model ─────────────────────────────────────────────────────────────
        final boolean useCustom = customModel != null && !customModel.isBlank();
        inputs.put( "model_file", useCustom ? customModel : builtinModel.modelFile() );

        // ── Core inference ────────────────────────────────────────────────────
        inputs.put( "conf",  conf );

             // ── Advanced ──────────────────────────────────────────────────────────
        inputs.put( "min_area", minArea );

        // ── GPU ───────────────────────────────────────────────────────────────
        inputs.put( "use_gpu", useGpu );

        return inputs;
    }


}
