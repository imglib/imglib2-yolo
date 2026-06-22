package net.imglib2.yolo;

import java.util.List;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.FolderOpener;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;

public class ImageSequenceUsage
{
	public static void main( final String[] args )
	{
		try
		{
			ImageJ.main( args );
			// https://www.kaggle.com/datasets/ultralytics/coco128
			final int n = 128; // to open
			final ImagePlus stack = FolderOpener.open( "samples/coco128/images/train2017", 640, 480, "count=" + n );
			stack.show();

			final Img< ARGBType > img = ImageJFunctions.wrap( stack );

			// Get messages about installing and processing
			final ApposeTaskListener listener = ApposeTaskListener.STD;

			// Specify the parameters for YOLO
			final YOLOSAHIParameters params = YOLOSAHIParameters.builder()
					.builtinModel( YOLOBuiltinModels.YOLO26L )
					.useSahi( false )
					.build();

			final List< List< YOLOResult > > output = YOLO.sahiDetectRGB( img, params, listener );
			final int totalObjects = output.stream().mapToInt( List::size ).sum();
			System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
			BasicUsage.showOutput( output, stack );
		}
		catch ( final Exception e )
		{
			e.printStackTrace();
		}
	}
}
