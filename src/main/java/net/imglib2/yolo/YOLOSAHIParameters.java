package net.imglib2.yolo;

import java.util.HashMap;
import java.util.Map;

import net.imglib2.appose.ShmImg;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class YOLOSAHIParameters extends YOLOMainParameters
{

    // ── SAHI ──────────────────────────────────────────────────────────────────

    public final int sliceHeight;

    public final int sliceWidth;

    public final double overlapHeightRatio;

    public final double overlapWidthRatio;

	public YOLOSAHIParameters(
			final YOLOBuiltinModels builtInModel,
			final String customModel,
			final double conf,
			final int minArea,
			final boolean useGpu,
			final int sliceHeight,
			final int sliceWidth,
			final double overlapHeightRatio,
			final double overlapWidthRatio
		)
	{
		super( builtInModel, customModel, conf, minArea, useGpu );
		this.sliceHeight = sliceHeight;
		this.sliceWidth = sliceWidth;
		this.overlapHeightRatio = overlapHeightRatio;
		this.overlapWidthRatio = overlapWidthRatio;
	}
	
	/** Default constructor with default values */
	public YOLOSAHIParameters() 
	{
		super();
		this.sliceHeight = 640;
		this.sliceWidth = 640;
		this.overlapHeightRatio = 0.2;
		this.overlapWidthRatio = 0.2;
	}


    /**
     * Builds the map passed to the Appose Python task.
     * Detections are returned via task.export() on the Python side,
     * so no output shared-memory image is needed.
     */
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input, final AxisInfo axisInfo )
    {
      
        final Map< String, Object > inputs = super.toApposeMap( input, axisInfo );

        // ── SAHI ──────────────────────────────────────────────────────────────
        inputs.put( "slice_height",                sliceHeight );
        inputs.put( "slice_width",                 sliceWidth );
        inputs.put( "overlap_height_ratio",        overlapHeightRatio );
        inputs.put( "overlap_width_ratio",         overlapWidthRatio );

        return inputs;
    }

}
