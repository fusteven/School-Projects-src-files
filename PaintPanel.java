package ca.utoronto.utm.paint;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.*;

/**
 * This class draws the shapes in various styles, colors using the commands 
 * from the paint model.
 * It captures the points and sends it to the Strategies.
 *  
 * @author csc207 student
 *
 */
class PaintPanel extends JPanel implements Observer, MouseMotionListener, MouseListener {

	private PaintModel model;
	private Point initialPoint;
	private StrategyInterface strategy;
	private ArrayList<ShapeDrawingCommand> shapeCommands; 

	private ArrayList<ThicknessSelectorCommand> finalThicknesses; 
	private ArrayList<ColorSelectorCommand> finalColors; 
	private ThicknessSelectorCommand draggedThickness; 
	private ColorSelectorCommand draggedColor;
	private BufferedImage image; 
	
	/**
	 * This constructor initializes the Strategies, default is circle, which could be updated 
	 * creates an ArrayList for thickness, colors, and shape drawing commands.
	 * 
	 * 
	 *@author csc207 student
	 * @param model
	 * @param view
	 * @param image
	 */
	public PaintPanel(PaintModel model, View view, BufferedImage image) {
		this.model = model;
		this.image = image;
		this.model.addObserver(this);
		this.addMouseListener(this);
		this.addMouseMotionListener(this);
		this.strategy = new CircleStrategy(this.model);
		this.shapeCommands = new ArrayList<>();
		this.finalThicknesses = new ArrayList<>();
		this.finalColors = new ArrayList<>();
		this.draggedThickness = new ThicknessSelectorCommand(1, model.DEFAULT);
		this.draggedColor = new ColorSelectorCommand(Color.BLACK);
	}
	/**
	 * This method draws the shape with the given commands. 
	 * 
	 * @author csc207 student
	 * 
	 */
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2d = (Graphics2D) g; // lets use the advanced api
		
		
        if (this.image != null)
            g.drawImage(image, 0, 0, this);
		shapeCommands = new ArrayList<ShapeDrawingCommand>();
		this.model.makeShapeCommands(shapeCommands);

		for(int i=0; i<this.shapeCommands.size(); i++) {

			this.finalThicknesses.get(i).execute(g2d);
			this.finalColors.get(i).execute(g2d);
			this.shapeCommands.get(i).execute(g2d);
		}
		
		if (this.model.getDraggedShape() != null) {
			this.draggedColor.execute(g2d);
			this.draggedThickness.execute(g2d);
			this.model.getDraggedShapeCommand().execute(g2d);	
		}
	}	
	
	// SETTERS
	
	public void addThicknessCommand(ThicknessSelectorCommand thickness) {
		this.finalThicknesses.add(thickness);
	}
	
	public void addColorCommand(ColorSelectorCommand color) {
		this.finalColors.add(color);
	}
	
	public void setDraggedThicknessCommand(ThicknessSelectorCommand thickness) {
		this.draggedThickness = thickness;
	}
	
	public void setDraggedColorCommand(ColorSelectorCommand color) {
		this.draggedColor = color;
	}
	public void setStrategy(StrategyInterface currentStrategy) {
		this.strategy = currentStrategy;}

	@Override
	public void update(Observable o, Object arg) {
		this.repaint();
	}
	
	// MOUSE-RELATED METHODS
	
	/**
	 * This method sends the initialPoint when the cursor first presses
	 * 
	 * @author csc207 student
	 */
	@Override
	public void mousePressed(MouseEvent e) {
		if (SwingUtilities.isRightMouseButton(e)) // Check if current Polyline needs to end
			if (this.strategy instanceof PolylineStrategy)
				((PolylineStrategy) this.strategy).setRightClicked(true);
		
		initialPoint = new Point(e.getX(), e.getY());
		this.strategy.setInitialPoint(initialPoint);
		this.strategy.setDraggingPoint(initialPoint);
		this.strategy.behaviour();
	}
	
	/**
	 * This method sends the dragging points when the cursor is dragging
	 * 
	 * @author csc207 student
	 */
	@Override
	public void mouseDragged(MouseEvent e) {
		Point pointDragged = new Point(e.getX(), e.getY());
		this.strategy.setDraggingPoint(pointDragged);
		this.strategy.behaviour();
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		this.strategy.setFinalPoint(new Point(e.getX(), e.getY()));
		this.strategy.setFinalShape(true);
		this.strategy.behaviour();
	}
	
	@Override
	public void mouseClicked(MouseEvent e) {}
	@Override
	public void mouseMoved(MouseEvent e) {}
	@Override
	public void mouseEntered(MouseEvent e) {}
	@Override
	public void mouseExited(MouseEvent e) {}
	
}
