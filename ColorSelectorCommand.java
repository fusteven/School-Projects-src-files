package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.Graphics2D;

/**
 * This class sets the color to be drawn with, chosen by the user. 
 * 
 * @author csc207 Student
 *
 */

public class ColorSelectorCommand implements ShapeModifierCommand{

	private Color drawColor;
	
	/**
	 * This constructor takes in the color to be drawn with from the color class.
	 * 
	 * @author csc207 Student
	 * @param drawColor
	 */
	public ColorSelectorCommand(Color drawColor) {
		this.drawColor = drawColor;
	}

	
	/**
	 * This methods sets the drawing color utilizing the Graphics2D's built-in method
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void execute(Graphics2D g) {
		g.setColor(drawColor);
	}

}
