package net.imglib2.yolo;

import net.imglib2.RealInterval;

public record YOLOResult( int id, int classId, String className, double score, double x1, double y1, double x2, double y2 ) implements RealInterval
{

	@Override
	public int numDimensions()
	{
		return 2;
	}

	public double width()
	{
		return x2 - x1;
	}

	public double height()
	{
		return y2 - y1;
	}

	@Override
	public double realMin( final int d )
	{
		return ( d == 0 ) ? x1 : y1;
	}

	@Override
	public double realMax( final int d )
	{
		return ( d == 0 ) ? x2 : y2;
	}
}
