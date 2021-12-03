package ca.utoronto.utm.paint;


/**
 * This class creates the object triangle with the initial point, endpoint, vertex and 
 * whether it is filled or not
 * The class Circle takes the super class Shape whether the circle is filled or not. 
 * 
 * @author csc207 student
 */
public class Triangle extends Shape{
	
	private Point initialPoint;
	private Point endPoint;
	private Point lastVertex;
	
	/**
	 * This method creates the necessary information for the triangle form the parameters
	 * 
	 * @param initialPoint
	 * @param endPoint
	 * @param vertex
	 * @param fill
	 */
	public Triangle(Point initialPoint, Point endPoint, Point vertex, boolean fill) {
		super(fill);
		this.initialPoint = initialPoint;
		this.endPoint = endPoint;
		this.lastVertex = vertex;
	}


	public Point getInitialPoint() {
		return initialPoint;
	}


	public Point getLastVertex() {
		return lastVertex;
	}


	public void setLastVertex(Point lastVertex) {
		this.lastVertex = lastVertex;
	}


	public void setInitialPoint(Point initialPoint) {
		this.initialPoint = initialPoint;
	}


	public Point getEndPoint() {
		return endPoint;
	}


	public void setEndPoint(Point endPoint) {
		this.endPoint = endPoint;
	}

	
}
