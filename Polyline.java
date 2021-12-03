package ca.utoronto.utm.paint;

import java.awt.Color;
import java.util.ArrayList;

/**
 * A Polyline object represents points distributed acroos some canvas connected in the order in
 * which they were added to said object
 * 
 * @author csc207 student
 *
 */
public class Polyline extends Shape implements LineTool {
	private ArrayList<Point> points = new ArrayList<Point>(); // Points making up Polyline
	
	/**
	 * Construct a Polyline object with starting Point point
	 * 
	 * @param point : Point by which to start the Polyline object
	 */
	public Polyline(Point point) {
		super(false);
		this.extend(point);
	}
	/**
	 * Construct a Polyline object with starting Point point with the provided color
	 * and thickness
	 * 
	 * @param point		: Point by which to start the Polyline object
	 * @param color		: Color of Polyline
	 * @param thickness : int >= 1
	 */
	public Polyline(Point point, Color color, int thickness) {
		super(false);  // fills is always set to false
		this.extend(point);
	}
	/**
	 * Construct a Polyline object represented by some list of Point objects
	 * 
	 * @param points : list of Point objects making up Polyline
	 */
	public Polyline(ArrayList<Point> points) {
		// Implicit call to no-parameter constructor of class Shape
		super(false);
		this.points = points;
	}
	/**
	 * Construct a Polyline object represented by some list of Point objects with the provided color
	 * and thickness
	 * 
	 * @param points	: list of Point objects making up Polyline
	 * @param color		: Color of Polyline
	 * @param thickness : int >= 1
	 */
	public Polyline(ArrayList<Point> points, Color color, int thickness) {
		super(false);  // fills is always set to false
		this.points = points;
	}
	
	/**
	 * Extend Polyline by Point p
	 * 
	 * @param p : The Point object by which to extend the shape being formed
	 */
	public void extend(Point p) {
		this.points.add(p);
	}
	
	/** Return Point at index
	 * 
	 * @param index : some int >= 0
	 * @return The point at the specified index
	 */
	public Point getPoint(int index) {
		return points.get(index);
	}
	
	/**
	 * Returns the number of Point objects in Polyline
	 * 
	 * @return : int >= 0
	 */
	public int length() {
		return points.size();
	}
	

	
}
