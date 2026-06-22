package net.imglib2.yolo;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * Utilities to convert and reshape images for YOLO detection.
 */
public class YOLOImgUtils
{

	/**
	 * Converts an ARGBType image to a 3-channel UnsignedByteType view by
	 * extracting the R, G, B components. The channel axis is appended as the
	 * last dimension.
	 * <p>
	 * Input [X, Y] → output [X, Y, 3] <br>
	 * Input [X, Y, Z] → output [X, Y, Z, 3]
	 * <p>
	 * The view is lazy: channel values are computed on access, so no extra
	 * memory is allocated until the view is copied.
	 *
	 * @param argb
	 *            the input ARGB image.
	 * @return a read-only view with 3 UnsignedByteType channels (R, G, B).
	 */
	public static RandomAccessibleInterval< UnsignedByteType > argbToRGBStack( final RandomAccessibleInterval< ARGBType > argb )
	{
		final RandomAccessibleInterval< UnsignedByteType > r = Converters.convert(
				argb,
				( in, out ) -> out.set( ARGBType.red( in.get() ) ),
				new UnsignedByteType() );

		final RandomAccessibleInterval< UnsignedByteType > g = Converters.convert(
				argb,
				( in, out ) -> out.set( ARGBType.green( in.get() ) ),
				new UnsignedByteType() );

		final RandomAccessibleInterval< UnsignedByteType > b = Converters.convert(
				argb,
				( in, out ) -> out.set( ARGBType.blue( in.get() ) ),
				new UnsignedByteType() );

		return Views.stack( r, g, b );
	}

	/**
	 * Converts a single-channel image to a 3-channel RGB stack by duplicating
	 * the channel three times. The channel axis is appended as the last
	 * dimension.
	 * <p>
	 * The input image should already be in the 0-255 range. For 16-bit or
	 * 32-bit images, rescale before calling this method.
	 *
	 * @param singleChannel
	 *            the input single-channel image.
	 * @return a read-only view with 3 UnsignedByteType channels.
	 */
	public static < T extends RealType< T > > RandomAccessibleInterval< UnsignedByteType > singleChannelToRGBStack( final RandomAccessibleInterval< T > singleChannel )
	{
		// Convert to 8-bit and clamps to 0-255
		final RandomAccessibleInterval< UnsignedByteType > converted = Converters.convert(
				singleChannel,
				( in, out ) -> out.set( ( int ) Math.round( in.getRealDouble() ) ),
				new UnsignedByteType() );
		return Views.stack( converted, converted, converted );
	}
}
