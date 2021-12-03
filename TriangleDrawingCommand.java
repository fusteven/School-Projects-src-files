package ca.utoronto.utm.paint;

import java.awt.Graphics2D;
import java.awt.Polygon;

/**
 * This class takes int the triangle class's necessary information which is used to
 * draw the triangle with 
 * the consideration of the Cursor Offset. 
 * 
 *@author csc207 Student
 */
public class TriangleDrawingCommand implements ShapeDrawingCommand{
	private Triangle triangle;
	private Point initialPoint;
	private Point endPoint;
	private Point lastVertex;
	private boolean isFilled;
	
	/**
	 * This constructor initializes the necessary information from the triangle class
	 * 
	 * @param triangle
	 * @author csc207 Student
	 */
	public TriangleDrawingCommand(Triangle triangle) {
		this.triangle = triangle;
		this.initialPoint = this.triangle.getInitialPoint();
		this.endPoint = this.triangle.getEndPoint();
		this.lastVertex = this.triangle.getLastVertex();
		this.isFilled = this.triangle.getFilled();
	}
	
	/**
	 * This method draws the actual triangle. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void execute(Graphics2D g) {
		int[] xs = {initialPoint.getX(), endPoint.getX(), lastVertex.getX()};
		int[] ys = {initialPoint.getY(), endPoint.getY(), lastVertex.getY()};
		Polygon p = new Polygon(xs, ys, 3);  
		
		if (!this.isFilled) {
			g.drawPolygon(p);
		} else {
			g.fillPolygon(p);
		}
		

		
	}
	
}
