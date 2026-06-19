package net.imglib2.yolo;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;

public class BasicUsage
{
	public static void main( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
		try
		{
			basicUsage( args );
		}
		catch ( final Exception e )
		{
			e.printStackTrace();
		}
	}

	public static void basicUsage( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
		// Demo preparation. We use IJ for this one.
		ImageJ.main( args );
		final ImagePlus imp = IJ.openImage( "samples/cycling001-1024x683.jpg" );
		imp.show();
		final Img< ARGBType > img = ImageJFunctions.wrap( imp );

		// Input
		final RandomAccessibleInterval< ARGBType > input = img;

		// Get messages about installing and processing
		final ApposeTaskListener listener = ApposeTaskListener.STD;

		// Specify the parameters for Cellpose 3
		final YOLOParameters params = YOLOParameters.builder()
				.builtinModel( YOLOBuiltinModels.YOLO26N )
				.useSahi( false )
				.build();

		final List< Map< String, Object > > output = YOLO.detectRGB( input, params, listener );

		System.out.println( "Detected " + output.size() + " objects" );
		for ( final Map< String, Object > detection : output )
		{
			System.out.println( detection );
		}

	}
}
