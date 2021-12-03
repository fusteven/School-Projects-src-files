package ca.utoronto.utm.paint;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

/**
 * This class creates a panel which visually displays the thickness level 
 * that user has chosen
 * 
 * @author csc207 student
 *
 */
public class ThicknessDisplayPanel extends JPanel {
	
	private int thickness;
	 
	/** 
	 * This constructor initializes a thickness display panel that
	 * takes in the current thickness and draws it.
	 * 
	 * @param thickness the chosen thickness to be displayed
	 */
	public ThicknessDisplayPanel(int thickness) {
		this.setBackground(Color.WHITE);
		this.setPreferredSize(new Dimension(30,40));
		this.thickness = thickness;
	}
	
	/** 
	 * This method draws a line with the current thickness
	 * 
	 * @param g a graphic component for drawing
	 */
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2d = (Graphics2D) g;
		g2d.setStroke(new BasicStroke(this.thickness));
		this.setPreferredSize(new Dimension(30,40+this.thickness));
		g2d.drawLine(0, this.getHeight()/2, this.getWidth(), this.getHeight()/2);
	}
	
	public void setThickness(int input) {
		this.thickness=input;
	}
	
	/** 
	 * This method refreshes the drawing panel
	 */
	public void displayThickness() {
		this.repaint();
	}
	
}
