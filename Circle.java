package ca.utoronto.utm.paint;

/**
 * This class creates the object circle with the center point, radius number, boolean filled
 * The class Circle takes the super class Shape whether the circle is filled or not. 
 * 
 * @author csc207 student
 */
public class Circle extends Shape{
	private Point centre; // Centre of the circle
	private int radius; // Radius of the circle
	
	/**
	 * This constructor creates the Circle with center point, length of radius, 
	 * filled or not 
	 * 
	 * @param centre 		: Point pointing to centre of circle
	 * @param radius 		: Radius of circle >= 0
	 * @param color  		: Color of circle

	 */
	public Circle(Point centre, int radius, boolean fill){
		super(fill);
		this.centre = centre;
		this.radius = radius;
	}
	
	/**
	 * This method returns the center point of the circle
	 * 
	 * @author csc207 student
	 */
	public Point getCentre() {
		return centre;
	}
	/** 
	 * This method sets the center point of the circle
	 * 
	 * @author csc207 student
	 * @param centre : Point pointing to centre of circle
	 */
	public void setCentre(Point centre) {
		this.centre = centre;
	}
	/** 
	 * This method returns the radius of the circle
	 * 
	 * @author csc207 student
	 */
	public int getRadius() {
		return radius;
	}
	/**
	 * This method sets the radius of the circle
	 * 
	 * @author csc207 student
	 * @param radius : Radius of the circle >= 0
	 */
	public void setRadius(int radius) {
		this.radius = radius;
	}

}
