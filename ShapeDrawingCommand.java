package ca.utoronto.utm.paint;
import java.awt.Graphics2D;

/**
 * This interface makes sure all other commands will be the execute, drawing, commands
 * 
 * @author csc207 students
 *
 */
public interface ShapeDrawingCommand {
	int CURSOR_OFFSET = 0;
	
	public void execute(Graphics2D g);
}
