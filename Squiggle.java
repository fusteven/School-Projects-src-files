package ca.utoronto.utm.paint;

import java.awt.Color;
import java.util.ArrayList;
/**
 * A Squiggle is a set of points connected in the order they are added to a Squiggle object
 * 
 * @author csc207 student
 *
 */
public class Squiggle extends Shape implements LineTool {
	private ArrayList<Point> squiggle = new ArrayList<Point>(); // Points making up Polyline
	
	/**
	 * Construct a Squiggle object with starting Point point
	 * 
	 * @param point : Point by which to start the Polyline object
	 */
//	public Squiggle(Point point) {
//		// Implicit call to no-parameter constructor of class Shape
//		this.extend(point);
//	}
	/**
	 * Construct a Squiggle object with starting Point point with the provided color
	 * and thickness
	 * 
	 * @param point		: Point by which to start the Squiggle object
	 * @param color		: Color of Squiggle
	 * @param thickness : int >= 1
	 */
	public Squiggle(Point point) {
		super(false);  // fills is always set to false
		this.extend(point);
	}
	/**
	 * Construct a Squiggle object represented by some list of Point objects
	 * 
	 * @param points : list of Point objects making up Squiggle
	 */
	public Squiggle(ArrayList<Point> squiggle) {
		// Implicit call to no-parameter constructor of class Shape
		super(false);
		this.squiggle = squiggle;
	}
	
	/**
	 * Extend Squiggle by Point p
	 * 
	 * @param p : The Point object by which to extend the shape being formed
	 */
	public void extend(Point point) {
		squiggle.add(point);
	}
	
	/** Return Point at index
	 * 
	 * @param index : some int >= 0
	 * @return The point at the specified index
	 */
	public Point getPoint(int index) {
		return squiggle.get(index);
	}
	
	/**
	 * Returns the number of Point objects in Squiggle
	 * 
	 * @return : int >= 0
	 */
	public ArrayList<Point> getSquiggle(){
		return this.squiggle;
	}
	@Override
	public int length() {
		return this.squiggle.size();
	}


	
	
	
}