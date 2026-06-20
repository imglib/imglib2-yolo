package net.imglib2.yolo;

import java.awt.Color;
import java.io.IOException;
import java.util.List;
import java.util.Map;

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

		final List< List< Map< String, Object > > > output = YOLO.detectRGB( input, params, listener );
		int totalObjects = output.stream().mapToInt( List::size ).sum();
		System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
		showOutput( output, imp );
	}

	private static void showOutput( final List< List< Map< String, Object > > > output, final ImagePlus imp )
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

		for ( final List< Map< String, Object > > plane : output )
		{
			for ( final Map< String, Object > detection : plane )
			{
				final double score = ( ( Number ) detection.get( "score" ) ).doubleValue();
				final int classId = ( int ) detection.get( "class_id" );
				final int id = ( int ) detection.get( "id" );
				final double x1 = ( ( Number ) detection.get( "x1" ) ).doubleValue();
				final double y1 = ( ( Number ) detection.get( "y1" ) ).doubleValue();
				final double x2 = ( ( Number ) detection.get( "x2" ) ).doubleValue();
				final double y2 = ( ( Number ) detection.get( "y2" ) ).doubleValue();
				final String className = ( String ) detection.get( "class_name" );

				final Roi roi = new Roi( x1, y1, x2 - x1, y2 - y1 );
				roi.setStrokeColor( get( classId ) );
				overlay.add( roi );

				final TextRoi textRoi = new TextRoi( x1, y1 - 20, id + ": " + className + " (" + String.format( "%.2f", score ) + ")" );
				textRoi.setFillColor( Color.BLACK );
				textRoi.setStrokeColor( get( classId ) );
				overlay.add( textRoi );
			}
		}

		imp.updateAndDraw();
		System.out.println( "Done." );
	}

	/**
	 * Get a random color, but fixed for a given n.
	 */
	private static final Color get(  final int n )
	{
		// Forbid dark color.
		final int r = ( ( ( 1 + n ) * 123 ) % 128 ) + 128;
		final int g = ( ( ( 2 + n ) * 456 ) % 128 ) + 128;
		final int b = ( ( ( 3 + n ) * 789 ) % 128 ) + 128;
		return new Color( r, g, b );
	}
}
