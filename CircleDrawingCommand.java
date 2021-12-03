package ca.utoronto.utm.paint;
import java.awt.Graphics2D;
/**
 * This class takes int the Circle class's necessary information which is used to
 * draw the circle with 
 * the consideration of the Cursor Offset. 
 * 
 *@author csc207 Student
 */

public class CircleDrawingCommand implements ShapeDrawingCommand{
	private Circle circle;
	private int r;
	private Point center;
	private int x;
	private int y;
	private final boolean isFilled;
	
	/**
	 * This constructor initializes the necessary information from the Circle class
	 * 
	 * @param circle
	 * @author csc207 Student
	 */
	public CircleDrawingCommand(Circle circle) {
		this.circle = circle;
		this.r = circle.getRadius();
		this.center = circle.getCentre();
		this.x = center.getX() - r;
		this.y = center.getY() - r;
		this.isFilled = this.circle.getFilled();
	}
	
	/**
	 * This method draws the actual circle. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	
	@Override
	public void execute(Graphics2D g) {
		if (this.isFilled) 
			g.fillOval(x + CURSOR_OFFSET, y + CURSOR_OFFSET, r*2, r*2);
		else 
			g.drawOval(x + CURSOR_OFFSET, y + CURSOR_OFFSET, r*2, r*2);
	}

}