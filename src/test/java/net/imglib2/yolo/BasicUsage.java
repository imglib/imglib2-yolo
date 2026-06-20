package net.imglib2.yolo;

import java.awt.Color;
import java.io.IOException;
import java.util.List;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.gui.TextRoi;
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

		// Specify the parameters for YOLO
		final YOLOParameters params = YOLOParameters.builder()
				.builtinModel( YOLOBuiltinModels.YOLO26N )
				.useSahi( true )
				.build();

		final List< List< YOLOResult > > output = YOLO.detectRGB( input, params, listener );
		final int totalObjects = output.stream().mapToInt( List::size ).sum();
		System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
		showOutput( output, imp );
	}

	private static void showOutput( final List< List< YOLOResult > > output, final ImagePlus imp )
	{
		// Prep overlay for output
		Overlay overlay = imp.getOverlay();
		if ( overlay == null )
		{
			overlay = new Overlay();
			imp.setOverlay( overlay );
		}
		else
		{
			overlay.clear();
		}

		for ( final List< YOLOResult > plane : output )
		{
			for ( final YOLOResult d : plane )
			{
				final Roi roi = Roi.create( d.x1(), d.y1(), d.width(), d.height() );
				roi.setStrokeColor( get( d.classId() ) );
				overlay.add( roi );

				final TextRoi textRoi = new TextRoi( d.x1(), d.y1() - 20, d.id() + ": " + d.className() + " (" + String.format( "%.2f", d.score() ) + ")" );
				textRoi.setFillColor( Color.BLACK );
				textRoi.setStrokeColor( get( d.classId() ) );
				overlay.add( textRoi );
			}
		}
		imp.updateAndDraw();
	}

	/**
	 * Get a random color, but fixed for a given n.
	 */
	private static final Color get(  final int n )
	{
		// Forbid dark color.
		final int r = ( ( n * 123 ) % 128 ) + 128;
		final int g = ( ( n * 456 ) % 128 ) + 128;
		final int b = ( ( n * 789 ) % 128 ) + 128;
		return new Color( r, g, b );
	}
}
