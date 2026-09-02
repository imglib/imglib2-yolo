package net.imglib2.yolo;
import ij.ImagePlus;
import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imglib2.img.ImagePlusAdapter;

public class UtilsForTest {
	public static AxisInfo getAxisInfo( final ImagePlus imp )
	{
		final ImgPlus<?> img = ImagePlusAdapter.wrapImgPlus( imp );
		final int x = img.dimensionIndex( Axes.X );
		final int y = img.dimensionIndex( Axes.Y );
		final int c = img.dimensionIndex( Axes.CHANNEL );
		final int z = img.dimensionIndex( Axes.Z );
		final int t = img.dimensionIndex( Axes.TIME );
		return new AxisInfo( x, y, c, z, t );
	}
}
