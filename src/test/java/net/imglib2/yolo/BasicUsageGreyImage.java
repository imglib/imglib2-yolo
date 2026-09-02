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
import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imagej.axis.CalibratedAxis;
import net.imagej.space.TypedSpace;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;

public class BasicUsageGreyImage
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
		final ImagePlus imp = IJ.openImage( "http://imagej.net/images/blobs.gif" );
		imp.show();
		final Img< UnsignedByteType > img = ImageJFunctions.wrap( imp );
		AxisInfo axisInfo = getAxisInfo( imp );  // get the axes information
		System.out.println(axisInfo.toString());
		
		// Get messages about installing and processing
		final ApposeTaskListener listener = ApposeTaskListener.STD;

		// Specify the parameters for YOLO
		final YOLOParameters params = new YOLOParameters();
		params.scale = 0.25;

		final RandomAccessibleInterval< UnsignedByteType > input = YOLOImgUtils.singleChannelToRGBStack( img );
		axisInfo = axisInfo.insertChannelDim(input.numDimensions()-1);
		
		final List< List< YOLOResult > > output = YOLOMain.detect( input, axisInfo, params, listener );
		final int totalObjects = output.stream().mapToInt( List::size ).sum();
		System.out.println( "Detected " + totalObjects + " objects in " + output.size() + " plane(s)" );
		showOutput( output, imp );
	}
	
	static AxisInfo getAxisInfo( final ImagePlus imp )
	{
		final ImgPlus<?> img = ImagePlusAdapter.wrapImgPlus( imp );
		final int x = img.dimensionIndex( Axes.X );
		final int y = img.dimensionIndex( Axes.Y );
		final int c = img.dimensionIndex( Axes.CHANNEL );
		final int z = img.dimensionIndex( Axes.Z );
		final int t = img.dimensionIndex( Axes.TIME );
		return new AxisInfo( x, y, c, z, t );
	}

	static void showOutput( final List< List< YOLOResult > > output, final ImagePlus imp )
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

		for ( int i = 0; i < output.size(); i++ )
		{
			final List< YOLOResult > plane = output.get( i );
			for ( final YOLOResult d : plane )
			{
				final Roi roi = Roi.create( d.x1(), d.y1(), d.width(), d.height() );
				roi.setStrokeColor( get( d.classId() ) );
				roi.setPosition( i + 1 );
				overlay.add( roi );

				final TextRoi textRoi = new TextRoi( d.x1(), d.y1() - 20, d.id() + ": " + d.className() + " (" + String.format( "%.2f", d.score() ) + ")" );
				textRoi.setFillColor( Color.BLACK );
				textRoi.setPosition( i + 1 );
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
