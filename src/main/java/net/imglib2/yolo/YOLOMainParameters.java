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
    }
    
    /** Constructor by values */
    public YOLOMainParameters( YOLOBuiltinModels builtinModel, String customModel, double conf, int minArea, boolean useGpu )
    {
        this.builtinModel                = builtinModel;
        this.customModel                 = customModel;
        this.conf                        = conf;
        this.minArea                     = minArea;
        this.useGpu                      = useGpu;
    }
    
    /**
     * Builds the map passed to the Appose Python task.
     * Detections are returned via task.export() on the Python side,
     * so no output shared-memory image is needed.
     */
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input )
    {
        final Map< String, Object > inputs = new HashMap<>();

        // ── Input image ───────────────────────────────────────────────────────
        inputs.put( "input", input.ndArray() );

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
