package net.imglib2.yolo;

import java.io.IOException;
import java.util.List;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.Typed;
import net.imglib2.appose.ShmImg;
import net.imglib2.converter.Converters;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * Main class to run YOLO detection with optional SAHI slicing from Java, using
 * Appose to manage Python environments and processes, and using ImgLib2 data
 * structures as input.
 */
public class YOLO
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

	/**
	 * Runs YOLO-SAHI detection on the given image and returns all detections.
	 * This method is suitable for image made of scalar pixel types, not RGB
	 * ones (ARGB type).
	 *
	 * @param <T>
	 *            the pixel type of the input image. Must be scalar.
	 * @param img
	 *            the input image.
	 * @param params
	 *            the YOLO-SAHI parameters.
	 * @param listener
	 *            receives progress and log messages.
	 * @return a list of list of detections. One list per plane.
	 * @throws BuildException
	 *             if building the Python environment fails.
	 * @throws IOException
	 *             if reading the Python scripts or environment specification
	 *             fails.
	 * @throws InterruptedException
	 *             if the Python process is interrupted.
	 * @throws TaskException
	 *             if executing the Python script fails.
	 */
	public static < T extends RealType< T > & NativeType< T > > List< List< YOLOResult > > sahiDetect(
			final RandomAccessibleInterval< T > img,
			final YOLOSAHIParameters params,
			final ApposeTaskListener listener ) throws BuildException, IOException, InterruptedException, TaskException
	{
		// Test if the image is truly scalar
		if ( img.getType() instanceof ARGBType )
			throw new IllegalArgumentException( "Input image must be scalar for non-RGB detection. Got " + ( ( Typed< ? > ) img ).getType().getClass().getSimpleName() );

		final String envName = getEnvName( params.useGpu );
		try (final ShmImg< UnsignedByteType > input = ShmImg.copyOf( singleChannelToRGBStack( img ) );
				YOLOSAHIRunner runner = new YOLOSAHIRunner(
				params,
				envName,
				listener,
				input ))
		{
			runner.init();
			return runner.run();
		}
	}

	/**
	 * Runs YOLO-SAHI detection on the given RGB image and returns all
	 * detections. This method is suitable for images made of ARGBType pixels,
	 * not scalar ones.
	 * 
	 * @param img
	 *            the input RGB image (ARGBType).
	 * @param params
	 *            the YOLO-SAHI parameters.
	 * @param listener
	 *            receives progress and log messages.
	 * @return a list of list of detections. One list per plane.
	 * @throws BuildException
	 *             if building the Python environment fails.
	 * @throws IOException
	 *             if reading the Python scripts or environment specification
	 *             fails.
	 * @throws InterruptedException
	 *             if the Python process is interrupted.
	 * @throws TaskException
	 *             if executing the Python script fails.
	 */
	public static List< List< YOLOResult > > sahiDetectRGB(
			final RandomAccessibleInterval< ARGBType > img,
			final YOLOSAHIParameters params,
			final ApposeTaskListener listener ) throws BuildException, IOException, InterruptedException, TaskException
	{
		// Test if the image is truly RGB
		if ( img.getType() instanceof ARGBType == false )
			throw new IllegalArgumentException( "Input image must be of type ARGBType for RGB detection. Got " + ( ( Typed< ? > ) img ).getType().getClass().getSimpleName() );

		final String envName = getEnvName( params.useGpu );
		try (final ShmImg< UnsignedByteType > input = ShmImg.copyOf( argbToRGBStack( img ) );
				YOLOSAHIRunner runner = new YOLOSAHIRunner(
						params,
						envName,
						listener,
						input ))
		{
			runner.init();
			return runner.run();
		}
	}

	/**
	 * Creates a YOLO runner for repeated inference calls with the same
	 * parameters and shared-memory placeholder.
	 * <p>
	 * Useful when processing many images with the same parameters, as the
	 * Python environment and model are initialized only once. Write new input
	 * data into the returned {@code input} ShmImg, then call
	 * {@link YOLOSAHIParameters#run()} to get the detections.
	 *
	 * @param <T>
	 *            the pixel type of the input image.
	 * @param params
	 *            the YOLO-SAHI parameters.
	 * @param listener
	 *            receives progress and log messages.
	 * @param input
	 *            shared-memory placeholder for the input image. Because YOLO
	 *            only works on RGB images, and because we can only pass scalar
	 *            images via shared memory, this must be a 3-channel, 8-bit
	 *            image where the planes represent the R, G and B channels. So
	 *            we expect the input to be [W, H, 3] with UnsignedByteType
	 *            pixels for single images or [N, W, H, 3] for a stack of N
	 *            images. Everything else will fail with an exception at the
	 *            Python level.
	 * @return a {@link YOLOSAHIRunner} ready to call
	 *         {@link YOLOSAHIRunner#init()} and then
	 *         {@link YOLOSAHIRunner#run()}.
	 */
	public static YOLOSAHIRunner yoloSAHIRunner(
			final YOLOSAHIParameters params,
			final ApposeTaskListener listener,
			final ShmImg< UnsignedByteType > input )
	{
		final String envName = getEnvName( params.useGpu );
		return new YOLOSAHIRunner(
				params,
				envName,
				listener,
				input );
	}

	private static String getEnvName( final boolean useGpu )
	{
		if ( !useGpu || !hasCUDA() )
			return "yolo-cpu";
		return "yolo-gpu";
	}

	private static boolean hasCUDA()
	{
		try
		{
			final ProcessBuilder pb = new ProcessBuilder( "nvidia-smi" );
			pb.redirectErrorStream( true );
			final Process process = pb.start();
			process.waitFor();
			return process.exitValue() == 0;
		}
		catch ( final IOException | InterruptedException e )
		{
			return false;
		}
	}

	private YOLO()
	{}
}
