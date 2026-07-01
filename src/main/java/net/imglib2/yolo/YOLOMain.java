package net.imglib2.yolo;

import java.io.IOException;
import java.util.List;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.appose.ShmImg;
import net.imglib2.type.numeric.integer.UnsignedByteType;

/**
 * Main class to run YOLO detection with optional SAHI slicing from Java, using
 * Appose to manage Python environments and processes, and using ImgLib2 data
 * structures as input.
 */
public class YOLOMain
{

	/**
	 * Runs YOLO-SAHI detection on the given RGB image, passed as a 3x 8-bit
	 * channels, and returns all detections.
	 * <p>
	 * YOLO only accepts RGB images, and we have to pass them as 3-channel
	 * UnsignedByteType images with Appose. In addition, the channel dimension
	 * must be the last one. So we accept:
	 * <ul>
	 * <li>single 2D images: [W, H, 3] UnsignedByteType</li>
	 * <li>multiple 2D images (stacks): [N, W, H, 3] UnsignedByteType</li>
	 * <ul>
	 * If you have more dimensions than this, flatten them first.
	 * 
	 * @param img
	 *            the input image.
	 * @param params
	 *            the YOLO-SAHI parameters.
	 * @param listener
	 *            receives progress and log messages.
	 * @return a list of list of detections. One list per plane.
	 * @see YOLOImgUtils YOLOImgUtils - methods to convert input images to the
	 *      required format.
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
	public static List< List< YOLOResult > > sahiDetect(
			final RandomAccessibleInterval< UnsignedByteType > img,
			final YOLOSAHIParameters params,
			final ApposeTaskListener listener ) throws BuildException, IOException, InterruptedException, TaskException
	{
		if ( img.numDimensions() > 4 || img.numDimensions() < 3 )
			throw new IllegalArgumentException( "The input image must have at least 3 dimensions." );
		if ( img.dimension( img.numDimensions() - 1 ) != 3 )
			throw new IllegalArgumentException( "The last dimension of the input image must be [W, H, 3] or [N, W, H, 3]." );

		final String envName = getEnvName( params.useGpu );
		try (final ShmImg< UnsignedByteType > input = ShmImg.copyOf( img );
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

	private YOLOMain()
	{}
}
