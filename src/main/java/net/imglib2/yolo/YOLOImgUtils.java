package net.imglib2.yolo;

import java.util.List;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.loops.LoopBuilder;
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
		final RandomAccessibleInterval< UnsignedByteType > converted = Converters.convert(
				singleChannel,
				( in, out ) -> out.set( Math.min( 255, Math.max( 0, ( int ) in.getRealDouble() ) ) ),
				new UnsignedByteType() );
		return Views.stack( converted, converted, converted );
	}

	/**
	 * Rescales a single-channel image to the 0-255 range and converts it to a
	 * 3-channel RGB stack.
	 * 
	 * @param <T>
	 *            the type of the input image, must extend RealType.
	 * @param img
	 *            the input single-channel image.
	 * @return a read-only view with 3 UnsignedByteType channels.
	 */
	public static < T extends RealType< T > > RandomAccessibleInterval< UnsignedByteType > rescale( final RandomAccessibleInterval< T > img )
	{
		// Compute min max.
		final double[] minMax = minMax( img );
		final double min = minMax[ 0 ];
		final double max = minMax[ 1 ];

		// Rescale to 0-255 range, still mono channel.
		final double scale = 255. / ( max - min );
		final RandomAccessibleInterval< T > rescaled = Converters.convert(
				img,
				( in, out ) -> out.setReal( ( in.getRealDouble() - min ) * scale ),
				img.getType().copy() );

		// To 3x 8-bit channels.
		return singleChannelToRGBStack( rescaled );
	}

	private static final < T extends RealType< T > > double[] minMax( final RandomAccessibleInterval< T > img )
	{
		final boolean multiThreaded = img.size() > 500_000;

		// Min & Max over all chunks in parallel.
		final List< double[] > minMaxes = LoopBuilder
				.setImages( img )
				.multiThreaded( multiThreaded )
				.forEachChunk( ( chunk ) -> {
					final double[] minMax = new double[] { Double.MAX_VALUE, -Double.MAX_VALUE };
					chunk.forEachPixel( p -> {
						final double value = p.getRealDouble();
						if ( value < minMax[ 0 ] )
							minMax[ 0 ] = value;
						if ( value > minMax[ 1 ] )
							minMax[ 1 ] = value;
					} );
					return minMax;
				} );

		// Reduce to global min & max.
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for ( final double[] minMax : minMaxes )
		{
			if ( minMax[ 0 ] < min )
				min = minMax[ 0 ];
			if ( minMax[ 1 ] > max )
				max = minMax[ 1 ];
		}
		return new double[] { min, max };
	}
}
