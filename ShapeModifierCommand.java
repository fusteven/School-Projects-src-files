package ca.utoronto.utm.paint;

import java.awt.Graphics2D;
/**
 * This interface makes sure all other modifier commands will be the execute(drawing) method
 * 
 * @author csc207 students
 *
 */
public interface ShapeModifierCommand {
	
	public void execute(Graphics2D g);
}
