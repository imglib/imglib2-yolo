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

	public YOLOParameters(
			final YOLOBuiltinModels builtInModel,
			final String customModel,
			final double scale,
			final double conf,
			final int minArea,
			final boolean useGpu			
		)
	{
		super( builtInModel, customModel, scale, conf, minArea, useGpu );
	}
	
	/** Default constructor with default values */
	public YOLOParameters() 
	{
		super();
	}

	
	@Override
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input, final AxisInfo axisInfo )
    {
		final Map< String, Object > inputs = super.toApposeMap( input, axisInfo );
		return inputs;
	}
    
 }
