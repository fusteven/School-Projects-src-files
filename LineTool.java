package ca.utoronto.utm.paint;

import java.awt.Color;

/**
 * LineTool describes the behaviour of tools that form shapes consisting of points connected by lines
 * 
 * @author csc207 student
 *
 */
public interface LineTool {
	
	/**
	 * Extend the shape craeted by LineTool by Point p
	 * 
	 * @param p : The Point object by which to extend the shape being formed
	 */
	void extend(Point p);
	/**
	 * Return Point at index
	 * 
	 * @param index : some int >= 0
	 * @return The point at the specified index
	 */
	Point getPoint(int index);
	/**
	 * Returns the number of Point objects in LineTool
	 * 
	 * @return : int >= 0
	 */
	int length();

}
