package net.imglib2.yolo;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
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
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Runs YOLO detection with optional SAHI slicing via Appose.
 * <p>
 * Detections are returned as a list of maps from {@link #run()}, each map
 * containing: {@code plane}, {@code id}, {@code class_id}, {@code class_name},
 * {@code score}, {@code x1}, {@code y1}, {@code x2}, {@code y2}.
 */
public class YOLORunner< T extends RealType< T > & NativeType< T > > implements AutoCloseable
{

	private final String envName;

	private final String pythonScriptPath;

	private final String pythonInitScriptPath;

	private final ApposeTaskListener listener;

	private final Map< String, Object > inputsParams;

	private Service python;

	private String yoloScript;

	/**
	 * Instantiates a YOLO+SAHI runner.
	 *
	 * @param params
	 *            the YOLO+SAHI parameters.
	 * @param pythonInitScriptPath
	 *            classpath resource path to the model init script
	 *            ({@code yolo_sahi_init.py}).
	 * @param pythonScriptPath
	 *            classpath resource path to the inference script
	 *            ({@code yolo_sahi.py}).
	 * @param envName
	 *            the pixi environment name to use ({@code yolo-gpu} or
	 *            {@code yolo-cpu}).
	 * @param listener
	 *            receives progress and log messages.
	 * @param input
	 *            shared-memory placeholder for the input image. The data
	 *            written here is read on every {@link #run()} call.
	 * @param inputAxisInfo
	 *            axis layout of the input image.
	 */
	YOLORunner(
			final YOLOParameters params,
			final String pythonInitScriptPath,
			final String pythonScriptPath,
			final String envName,
			final ApposeTaskListener listener,
			final ShmImg< T > input )
	{
		this.pythonScriptPath = pythonScriptPath;
		this.pythonInitScriptPath = pythonInitScriptPath;
		this.envName = envName;
		this.listener = listener;
		this.inputsParams = params.toApposeMap( input );
	}

	/**
	 * Runs YOLO+SAHI detection on the image currently written in the input
	 * placeholder.
	 *
	 * @return a list of planes, each plane being a list of detections.
	 *         Each detection is a map with keys: {@code id}, {@code class_id},
	 *         {@code class_name}, {@code score}, {@code x1}, {@code y1},
	 *         {@code x2}, {@code y2}. Returns an empty list if no objects
	 *         were detected.
	 * @throws InterruptedException
	 *             if the thread is interrupted while waiting for the Python
	 *             script to finish.
	 * @throws TaskException
	 *             if executing the Python script fails.
	 */
	@SuppressWarnings( "unchecked" )
	public List< List< Map< String, Object > > > run() throws InterruptedException, TaskException
	{
		final Task task = python.task( yoloScript, inputsParams );

		final long start = System.currentTimeMillis();
		task.listen( listener.taskListener() );
		task.start();
		task.waitFor();

		if ( task.status != TaskStatus.COMPLETE )
			throw new RuntimeException( "Python script failed with error: " + task.error );

		final long end = System.currentTimeMillis();
		listener.message( "YOLO+SAHI inference done in " + ( end - start ) / 1000. + " s" );

		// Detections are returned via task.export( detections=... ) in Python.
		// Returns a list of lists: one list per plane.
		final Object detections = task.outputs.get( "detections" );
		if ( detections == null )
			return Collections.emptyList();
		return ( List< List< Map< String, Object > > > ) detections;
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
				.environment( envName )
				.build();

		// Start the Python worker, pre-loading the shared utility module.
		final String utilsScript = IOUtils.toString(
				YOLORunner.class.getResource( "/yolo_utils.py" ),
				StandardCharsets.UTF_8 );
		this.python = env.python().init( utilsScript );

		// Run the model init script (loads YOLO via SAHI, exports 'model').
		final String yoloInitScript = IOUtils.toString(
				YOLORunner.class.getResource( pythonInitScriptPath ),
				StandardCharsets.UTF_8 );
		final Task task = python.task( yoloInitScript, inputsParams );

		final long start = System.currentTimeMillis();
		task.listen( listener.taskListener() );
		task.start();
		task.waitFor();

		if ( task.status != TaskStatus.COMPLETE )
			throw new RuntimeException( "Python init script failed with error: " + task.error );

		final long end = System.currentTimeMillis();
		listener.message( "YOLO+SAHI initialisation done in " + ( end - start ) / 1000. + " s" );

		// Cache the inference script for repeated run() calls.
		this.yoloScript = IOUtils.toString(
				YOLORunner.class.getResource( pythonScriptPath ),
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
		final URL pixiFile = YOLORunner.class.getResource( "/pixi.toml" );
		return IOUtils.toString( pixiFile, StandardCharsets.UTF_8 );
	}
}
