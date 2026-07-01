package net.imglib2.yolo;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;

import net.imglib2.appose.ShmImg;
import net.imglib2.cellpose.Cellpose;
import net.imglib2.type.numeric.integer.UnsignedByteType;

/**
 * Runs YOLO detection with SAHI slicing via Appose.
 */
public class YOLOSAHIRunner implements AutoCloseable
{

	private static final String INIT_SCRIPT_PATH = "/yolo_sahi_init.py";

	private static final String RUN_SCRIPT_PATH = "/yolo_sahi.py";

	private static final String UTILS_SCRIPT_PATH = "/yolo_utils.py";

	private static final String PIXI_TOML_PATH = "/pixi_sahi.toml";

	private final String envName;

	private final ApposeTaskListener listener;

	private final Map< String, Object > inputsParams;

	private Service python;

	private String yoloScript;

	/**
	 * Instantiates a YOLO-SAHI runner.
	 *
	 * @param params
	 *            the YOLO-SAHI parameters.
	 * @param envName
	 *            the pixi environment name to use ({@code yolo-gpu} or
	 *            {@code yolo-cpu}).
	 * @param listener
	 *            receives progress and log messages.
	 * @param input
	 *            input image placehoder. Because YOLO only works on RGB images,
	 *            and because we can only pass scalar images via shared memory,
	 *            this must be a 3-channel, 8-bit image where the planes
	 *            represent the R, G and B channels. So we expect the input to
	 *            be [W, H, 3] with UnsignedByteType pixels for single images or
	 *            [N, W, H, 3] for a stack of N images. Everything else will
	 *            fail with an exception at the Python level.
	 */
	YOLOSAHIRunner(
			final YOLOSAHIParameters params,
			final String envName,
			final ApposeTaskListener listener,
			final ShmImg< UnsignedByteType > input )
	{
		this.envName = envName;
		this.listener = listener;
		this.inputsParams = params.toApposeMap( input );
	}

	/**
	 * Runs YOLO-SAHI detection on the image currently written in the input
	 * placeholder.
	 *
	 * @return a list of list of detections: one list per plane. Each detection
	 *         contains the bounding box coordinates, class ID and name, and
	 *         confidence score.
	 * @throws InterruptedException
	 *             if the thread is interrupted while waiting for the Python
	 *             script to finish.
	 * @throws TaskException
	 *             if executing the Python script fails.
	 */
	@SuppressWarnings( "unchecked" )
	public List< List< YOLOResult > > run() throws InterruptedException, TaskException
	{
		final Task task = python.task( yoloScript, inputsParams );

		final long start = System.currentTimeMillis();
		task.listen( listener.taskListener() );
		task.start();
		task.waitFor();

		if ( task.status != TaskStatus.COMPLETE )
			throw new RuntimeException( "Python script failed with error: " + task.error );

		final long end = System.currentTimeMillis();
		listener.message( "YOLO-SAHI inference done in " + ( end - start ) / 1000. + " s" );

		// Detections are returned via task.export( detections=... ) in Python.
		// Returns a list of lists: one list per plane.
		final Object detections = task.outputs.get( "detections" );
		if ( detections == null )
			return Collections.emptyList();

		return convertResults( ( List< List< Map< String, Object > > > ) detections );
	}

	/**
	 * Initialises the runner: builds the pixi environment, starts the Python
	 * worker, and loads the YOLO model. Must be called once before
	 * {@link #run()}.
	 *
	 * @throws IOException
	 *             if a script resource or the pixi.toml cannot be read.
	 * @throws BuildException
	 *             if building the pixi environment fails.
	 * @throws InterruptedException
	 *             if the thread is interrupted during model initialisation.
	 * @throws TaskException
	 *             if the model init script fails.
	 */
	public void init() throws IOException, BuildException, InterruptedException, TaskException
	{
		// Build the pixi environment.
		final Environment env = Appose
				.pixi()
				.content( pixiEnv() )
				.subscribeProgress( listener.progressListener() )
				.subscribeOutput( listener.outputListener() )
				.subscribeError( listener.errorListener() )
				.build();

		// Start the Python worker, pre-loading the shared utility module.
		final String utilsScript = IOUtils.toString(
				YOLOSAHIRunner.class.getResource( UTILS_SCRIPT_PATH ),
				StandardCharsets.UTF_8 );
		
		this.python = env.activate(envName).python().init( utilsScript );

		// Run the model init script (loads YOLO via SAHI, exports 'model').
		final String yoloInitScript = IOUtils.toString(
				YOLOSAHIRunner.class.getResource( INIT_SCRIPT_PATH ),
				StandardCharsets.UTF_8 );
		
		final Task task = python.task( yoloInitScript, inputsParams );

		final long start = System.currentTimeMillis();
		task.listen( listener.taskListener() );
		task.start();
		task.waitFor();

		if ( task.status != TaskStatus.COMPLETE )
			throw new RuntimeException( "Python init script failed with error: " + task.error );

		final long end = System.currentTimeMillis();
		listener.message( "YOLO-SAHI: Initialisation done in " + ( end - start ) / 1000. + " s" );

		// Cache the inference script for repeated run() calls.
		this.yoloScript = IOUtils.toString(
				YOLOSAHIRunner.class.getResource( RUN_SCRIPT_PATH ),
				StandardCharsets.UTF_8 );
	}

	@Override
	public void close()
	{
		python.close();
	}

	/**
	 * Reads the {@code pixi.toml} bundled as a classpath resource.
	 *
	 * @return the file content as a string.
	 * @throws IOException
	 *             if the resource cannot be read.
	 */
	public static String pixiEnv() throws IOException
	{
		final URL pixiFile = YOLOSAHIRunner.class.getResource( PIXI_TOML_PATH );
		return IOUtils.toString( pixiFile, StandardCharsets.UTF_8 );
	}

	private List< List< YOLOResult > > convertResults( final List< List< Map< String, Object > > > detections )
	{
		final List< List< YOLOResult > > results = new ArrayList<>( detections.size() );
		for ( final List< Map< String, Object > > plane : detections )
		{
			final List< YOLOResult > planeResults = new ArrayList<>( plane.size() );
			for ( final Map< String, Object > detection : plane )
			{
				final int id = ( int ) detection.get( "id" );
				final int classId = ( int ) detection.get( "class_id" );
				final String className = ( String ) detection.get( "class_name" );
				final double score = ( ( Number ) detection.get( "score" ) ).doubleValue();
				final double x1 = (( Number ) detection.get( "x1" )).doubleValue();
				final double y1 = (( Number ) detection.get( "y1" )).doubleValue();
				final double x2 = (( Number ) detection.get( "x2" )).doubleValue();
				final double y2 = (( Number ) detection.get( "y2" )).doubleValue();
				planeResults.add( new YOLOResult( id, classId, className, score, x1, y1, x2, y2 ) );
			}
			results.add( planeResults );
		}
		return results;
	}
}
