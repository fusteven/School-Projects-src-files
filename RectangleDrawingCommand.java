package ca.utoronto.utm.paint;

import java.awt.Graphics2D;
/**
 * This class takes the rectangle class's necessary information which is used to
 * draw the rectangle with the consideration of the Cursor Offset. 
 * 
 *@author csc207 Student
 */
public class RectangleDrawingCommand implements ShapeDrawingCommand{
	private Rectangle rect;
	private Point start;
	private boolean isFilled;
	
	/**
	 * This constructor initializes the necessary information from the rectangle class
	 * 
	 * @param rectangle
	 * @author csc207 Student
	 */
	public RectangleDrawingCommand(Rectangle rectangle) {
		this.rect = rectangle;
		this.start = rect.startingPoint();
		this.isFilled = this.rect.getFilled();
	}
	
	/**
	 * This method draws the actual rectangle. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void execute(Graphics2D g) {
		if (this.isFilled) 
			g.fillRect(start.getX() + ShapeDrawingCommand.CURSOR_OFFSET,
					   start.getY() + ShapeDrawingCommand.CURSOR_OFFSET,
					   rect.getWidth(),
					   rect.getLength());
		else 
			g.drawRect(start.getX() + ShapeDrawingCommand.CURSOR_OFFSET,
					   start.getY() + ShapeDrawingCommand.CURSOR_OFFSET,
					   rect.getWidth(),
					   rect.getLength());
	}

}