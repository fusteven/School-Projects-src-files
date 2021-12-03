package ca.utoronto.utm.paint;

import java.awt.Graphics2D;

/**
 * This class takes int the Polyline class's necessary information which is used to
 * draw the polyline with 
 * the consideration of the Cursor Offset. 
 * 
 *@author csc207 Student
 */
public class PolyLineDrawingCommand implements ShapeDrawingCommand{
	private Polyline polyline;
	
	/**
	 * This constructor initializes the necessary information from the Polyline class
	 * 
	 * @param polyline
	 * @author csc207 Student
	 */
	public PolyLineDrawingCommand(Polyline polyline) {
		this.polyline = polyline;
	}
	
	/**
	 * This method draws the actual polyline. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void execute(Graphics2D g) {	
		for (int i = 0; i < this.polyline.length() - 1 ; i++) {
			Point p1 = this.polyline.getPoint(i);
			Point p2 = this.polyline.getPoint(i+1);
			g.drawLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
		}
	}

 
}