package ca.utoronto.utm.paint;

import java.awt.Color;

/**
 * This method handles the rectangle that is created. It has the point where the button first clicked,
 * the length and width values, color for the shape, whether the rectangle is filled or not
 * and finally the thickness of the rectangle. 
 * 
 * @author csc207 student
 *
 */
public class Rectangle extends Shape{ 
	private Point start; // Top-left corner of Rectangle
	private int width; // Horizontal dimension of Rectangle
	private int length; // Vertical dimension of Rectangle
	
	/**
	 * This constructor takes in the starting point, length and width values, color,
	 * whether it is filled or not and the thickness values and creates the rectangle. 
	 * 
	 * @author csc207 student
	 * @param start			: Top-left corner of Rectangle
	 * @param length		: Vertical dimension of Rectangle
	 * @param width			: Horizontal dimension of Rectangle
	 * 
	 */
	public Rectangle (Point start, int length, int width, boolean fill){
		super(fill);
		this.start = start;
		this.width = width;
		this.length = length;
	}
	/**
	 * Return top-left corner of Rectangle
	 * 
	 * @return Point pointing to top-left corner of Rectangle
	 */
	public Point startingPoint() { 
		return this.start;
	}
	/**
	 * Return horizontal dimension of rectangle
	 * 
	 * @return int width of Ractangle
	 */
	public int getWidth() { 
		return this.width;
	}
	/**
	 * Return vertical dimension of rectangle
	 * 
	 * @return int length of Ractangle
	 */
	public int getLength() {
		return this.length;
	}
	/**
	 * Set horizontal dimension of rectangle
	 * 
	 * @param width : int width of Ractangle
	 */
	public void setWidth(int width) {
		this.width = width;
	}
	/**
	 * Set vertical dimension of rectangle
	 * 
	 * @param length : int length of Rectangle
	 */
	public void setLength(int length) {
		this.length = length ;
	}

}