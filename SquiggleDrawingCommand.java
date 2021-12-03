package ca.utoronto.utm.paint;

import java.awt.Graphics2D;

/**
 * This class takes int the squiggle class's necessary information which is used to
 * draw the squiggle with the consideration of the Cursor Offset. 
 * 
 *@author csc207 Student
 */
public class SquiggleDrawingCommand implements ShapeDrawingCommand{

	private Squiggle squiggle;
	
	
	/**
	 * This constructor initializes the necessary information from the squiggle class
	 * 
	 * @param squiggle
	 * @author csc207 Student
	 */
	
	public SquiggleDrawingCommand(Squiggle squiggle) {
		this.squiggle = squiggle;
	}
	
	/**
	 * This method draws the actual squiggle. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	
	@Override
	public void execute(Graphics2D g) {
		for (int i=0; i < this.squiggle.length() - 1; i++) {
			Point p1 = this.squiggle.getPoint(i);
			Point p2 = this.squiggle.getPoint(i+1);
			g.drawLine(p1.getX(), p1.getY(), p2.getX(), p2.getY());
		}
	}

}
