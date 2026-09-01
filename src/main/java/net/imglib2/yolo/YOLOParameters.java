package net.imglib2.yolo;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.imglib2.appose.ShmImg;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class YOLOParameters extends YOLOMainParameters
{
	public double scale; // scaling of the image to determine imgsz
	
	public YOLOParameters(
			final YOLOBuiltinModels builtInModel,
			final String customModel,
			final double conf,
			final int minArea,
			final boolean useGpu,
			final double scale
		)
	{
		super( builtInModel, customModel, conf, minArea, useGpu );
		this.scale = scale;
	}
	
	/** Default constructor with default values */
	public YOLOParameters() 
	{
		super();
		this.scale = 1.0;
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
	
	@Override
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input )
    {
		final Map< String, Object > inputs = super.toApposeMap( input );
		final boolean isBuiltInModel = customModel == null || customModel.equals( "" );
		
		final int imgsz = calculate_imgsz( (int) input.dimension(0), (int) input.dimension(1) ); // checker that dimensions are width and height 
		inputs.put( "imgsz", imgsz );
		return inputs;
	}
    
 }
