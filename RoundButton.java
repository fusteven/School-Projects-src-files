package ca.utoronto.utm.paint;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;


/**
 * This class creates the buttons for the color selector panel. It sets color of the button 
 * to a specific color which was set in ColorSelectorPanel class
 * 
 * @author csc207 student
 *
 */
public class RoundButton extends JButton {
	
	private Color color;
	private String colorName;
	 
	/**
	 * This constructor takes in the name of the color and the actual color code. It sets each of the button in the panel.
	 * 
	 * @param colorName a string that represents the color
	 * @param color a color object to be passed to the drawing
	 */
	public RoundButton(String colorName, Color color) {
		super(colorName);
		this.color = color;
		Dimension size = getPreferredSize();
		size.width = size.height = Math.max(size.width, size.height);
		setPreferredSize(size);
		setContentAreaFilled(false);
		setFocusPainted(false);
	}
	

	/**
	 * The method paints a circle with the current color.
	 * 
	 * @param g a graphic component for drawing
	 */
	protected void paintComponent(Graphics g) {
	     g.setColor(this.color);
	     g.fillOval(0, 0, 60, 60);
	     super.paintComponent(g);
	}
	      
	/**
	 * The method paints a border for the circle drawn.
	 * 
	 * @param g a graphic component for drawing
	 */
	protected void paintBorder(Graphics g) {
	     g.drawOval(0, 0, 0, 0);
	}
	
	java.awt.Shape shape;
	/** 
	 * This method checks if the x and y values are within the round button
	 * 
	 * @param x x coordinate of the mouse click
	 * @param y y coordinate of the mouse click
	 */
	public boolean contains(int x, int y) {
	     if (shape == null || !shape.getBounds().equals(getBounds())) {
	          shape = new Ellipse2D.Float(0, 0, getWidth(), getHeight());
	     }
	     return shape.contains(x, y);
	}
	
	public String getColorName() {
		return this.colorName;
	}
}
