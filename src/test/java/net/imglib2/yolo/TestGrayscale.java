package net.imglib2.yolo;

import java.io.IOException;
import java.util.List;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class TestGrayscale
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

	public static < T extends RealType< T > & NativeType< T > > void basicUsage( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
//		final String sampleImagePath = "samples/donut-bw-8bit.tif";
//		final String sampleImagePath = "samples/donut-bw-16bit-8bit-range.tif";
		final String sampleImagePath = "samples/donut-bw-16bit-16bit-range.tif";

		// Demo preparation. We use IJ for this one.
		ImageJ.main( args );
		final ImagePlus imp = IJ.openImage( sampleImagePath );
		imp.show();
		final Img< T > img = ImageJFunctions.wrap( imp );

		// Input
		final RandomAccessibleInterval< T > input = img;

		// Get messages about installing and processing
		final ApposeTaskListener listener = ApposeTaskListener.STD;

		// Specify the parameters for YOLO
		final YOLOSAHIParameters params = YOLOSAHIParameters.builder()
				.builtinModel( YOLOBuiltinModels.YOLO26L )
				.useSahi( true )
				.build();

		final List< List< YOLOResult > > output = YOLO.sahiDetect( input, params, listener );
		final int totalObjects = output.stream().mapToInt( List::size ).sum();
		System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
		BasicUsage.showOutput( output, imp );
	}
}
