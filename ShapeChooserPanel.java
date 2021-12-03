package ca.utoronto.utm.paint;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * This class creates the panel where the user can select what kind of shape he or she desires. 
 * The button icons are from a filename.
 * 
 * @author csc207 student
 *
 */
class ShapeChooserPanel extends JPanel implements ActionListener {
	
	private View view; 
	private Color buttonColor =  new Color(225, 225, 225);
	private JButton[] shapeButtons;
	
	/**
	 * This constructor takes in the View class and adds a JPanel to it which allows the user to select the desired
	 * shape. 
	 * 
	 * @author csc207 student
	 * @param view
	 */
	public ShapeChooserPanel(View view) {
		this.view = view;
		this.setCursor( new Cursor(Cursor.HAND_CURSOR));
		
		String[] buttonLabels = { "Circle", "Rectangle", "Square", "Triangle", "Squiggle", "Polyline"};
		ImageIcon[] buttonIcons = {new ImageIcon("images/circle.png"),
								   new ImageIcon("images/rectangle.png"),
								   new ImageIcon("images/square.png"),
								   new ImageIcon("images/triangle.png"),
								   new ImageIcon("images/squiggle.png"), 
								   new ImageIcon("images/polyline.png")};
		String[] buttonTips = {"Draw circles by positioning the center of circle on screen",
							   "Draw rectangles by positioning the top-left corner",
							   "Draw squares by positioning the top-left corner",
							   "Draw right triangles by positioning the hypotenuse",
							   "Freehand drawing by mouse dragging",
							   "Draw polylines by mouse click, right-click to end current polyline",
								};
		
		this.setLayout(new GridLayout(buttonLabels.length, 3));
		this.setBackground(Color.WHITE);
		
		shapeButtons = new JButton[buttonLabels.length];
		for (int i=0; i < shapeButtons.length; i++) {
			shapeButtons[i] = new JButton(buttonLabels[i], buttonIcons[i]);
			shapeButtons[i].setToolTipText("<html><p width=\"130px\">"+buttonTips[i]+"</p></html>");
			shapeButtons[i].setVerticalTextPosition(SwingConstants.BOTTOM);
			shapeButtons[i].setHorizontalTextPosition(SwingConstants.CENTER);
			shapeButtons[i].setBackground(buttonColor);
			shapeButtons[i].setBorderPainted(false);
			shapeButtons[i].setFocusPainted(false);
			shapeButtons[i].setRolloverEnabled(false);
			
			shapeButtons[i].addActionListener(this);
			this.add(shapeButtons[i]);
		}
		// Initially, button at index 0 will already be selected
		shapeButtons[0].setBackground(buttonColor.darker());
		this.changeShapeCursor(shapeButtons[0].getActionCommand());
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		JButton buttonPressed = (JButton) e.getSource();
		
		// Reset color of any previously clicked button
		for (JButton button : this.shapeButtons) {
			button.setBackground(buttonColor);
		}
		buttonPressed.setBackground(buttonColor.darker());  // Make pressed button darker
		
		this.changeShapeCursor(e.getActionCommand());  // Set the cursor
		
		// Set the program mode
		StrategyInterface currentStrategy = 
						  StrategyFactory.getInstance().createStrategy(e.getActionCommand());
		this.view.getPaintPanel().setStrategy(currentStrategy);
	}
	
	private void changeShapeCursor(String shape) {
		Toolkit tools = Toolkit.getDefaultToolkit();
		Cursor cursor;
		Image cursorImg;
		
		if (shape.equals("Squiggle")) {
			cursorImg = tools.createImage("images/cursors/pencil.png");
			java.awt.Point cursorPoint0 = new java.awt.Point(0,0);
			cursor = tools.createCustomCursor(cursorImg, cursorPoint0, " Pencil");
			this.view.setCursor(cursor);
		} else if (shape.equals("Circle")){ // Shape is "Circle" || "Rectangle" || "Square"
			java.awt.Point cursorPoint1 = new java.awt.Point(19,18);
			cursorImg = tools.createImage("images/cursors/circleSelectionCursor.png");
			cursor = tools.createCustomCursor(cursorImg, cursorPoint1, "Pencil");
			this.view.setCursor(cursor);
		} else if (shape.equals("Rectangle")){
			java.awt.Point cursorPoint2 = new java.awt.Point(15,22);
			cursorImg = tools.createImage("images/cursors/rectangle-tool.png");
			cursor = tools.createCustomCursor(cursorImg, cursorPoint2, "Pencil");
			this.view.setCursor(cursor);
		} else  if (shape.equals("Square")){ 
				java.awt.Point cursorPoint3 = new java.awt.Point(18,20);
				cursorImg = tools.createImage("images/cursors/SquareCursor.png");
				cursor = tools.createCustomCursor(cursorImg, cursorPoint3, "Pencil");
				this.view.setCursor(cursor);
		} else  if (shape.equals("Polyline")){ 
			java.awt.Point cursorPoint4 = new java.awt.Point(0,0);
			cursorImg = tools.createImage("images/cursors/PolylineCursor.png");
			cursor = tools.createCustomCursor(cursorImg, cursorPoint4, "Pencil");
			this.view.setCursor(cursor);
		} else  if (shape.equals("Triangle")){ 
			java.awt.Point cursorPoint5 = new java.awt.Point(27,27);
			cursorImg = tools.createImage("images/cursors/right-triangle.png");
			cursor = tools.createCustomCursor(cursorImg, cursorPoint5, "Pencil");
			this.view.setCursor(cursor);
		}
	}
	
}
