package net.imglib2.yolo;

import java.io.IOException;
import java.util.List;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.FolderOpener;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class TestGrayscale
{

	public static void main( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
		try
		{
//			singleImage( args );
			imageSequence( args );
		}
		catch ( final Exception e )
		{
			e.printStackTrace();
		}
	}

	public static < T extends RealType< T > & NativeType< T > > void imageSequence( final String[] args ) throws BuildException, IOException, InterruptedException, TaskException
	{
		ImageJ.main( args );
		final String filter = "donut-bw-16bit";
		final ImagePlus stack = FolderOpener.open( "samples/", 640, 480, "filter=" + filter );
		stack.show();

		AxisInfo axisInfo = UtilsForTest.getAxisInfo( stack );  // get the axes information
		final Img< T > img = ImageJFunctions.wrap( stack );

		// Get messages about installing and processing
		final ApposeTaskListener listener = ApposeTaskListener.STD;

		// Specify the parameters for YOLO
		final YOLOSAHIParameters params = new YOLOSAHIParameters();
		params.builtinModel = YOLOBuiltinModels.YOLO26L;
				
		final RandomAccessibleInterval< UnsignedByteType > input = YOLOImgUtils.rescale( img );
		axisInfo = axisInfo.insertChannelDim(input.numDimensions()-1);
		final List< List< YOLOResult > > output = YOLOMain.sahiDetect( input, axisInfo, params, listener );
		final int totalObjects = output.stream().mapToInt( List::size ).sum();
		System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
		BasicUsage.showOutput( output, stack );
	}

}
