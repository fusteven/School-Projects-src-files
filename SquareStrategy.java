package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.Observable;

/**
* This class handles all the necessary  calculations, regardless, if
* the mouse is dragging or the mouse is released. 
* 
* @author csc207 Student
*
*/
	
public class SquareStrategy extends Observable implements StrategyInterface  {


	public Point initialPoint;
	public Point draggingPoint;
	public PaintModel model = new PaintModel();
	public boolean FinalShape = false;
	private Point FinalPoint;
	private Rectangle rectangle;
	
	
	/**
	 * This constructor takes in the model and adds the square shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public SquareStrategy(PaintModel model) {
		this.model = model;
		System.out.println(this.model);
	}
	
	/**
	 * This method is called to do the necessary calculations with the initial, dragging
	 * and final mouse click positions for the square. 
	 * 
	 * @author csc207 Student
	 */
	public void behaviour() {
		if (initialPoint != null && draggingPoint != null) {
			Point default_initial = new Point(initialPoint.getX(), initialPoint.getY());
			int length_x = Math.abs(draggingPoint.getX() - initialPoint.getX());
			int length_y = Math.abs(draggingPoint.getY() - initialPoint.getY());
			if (length_x < length_y)
				length_x = length_y;
			int square_length = length_x; // determine the larger length
			if (draggingPoint.getX() < initialPoint.getX())
				default_initial.setX(initialPoint.getX() - square_length);
			if (draggingPoint.getY() < initialPoint.getY())
				default_initial.setY(initialPoint.getY() - square_length);
			model.setDraggedShape((Shape) new Rectangle(default_initial, square_length, square_length, this.model.getIsFilled()));
		}
		if (FinalShape != false && FinalPoint != null) {
			Point squarePosition = new Point(initialPoint.getX(),initialPoint.getY());	
			int length_x = Math.abs(FinalPoint.getX() - initialPoint.getX());
			int length_y = Math.abs(FinalPoint.getY() - initialPoint.getY());
			if (length_x < length_y) {
				length_x = length_y;
			}
			int square_length = length_x; // determine the larger length			
			if (FinalPoint.getX() < initialPoint.getX()) {
				squarePosition.setX(initialPoint.getX() - square_length);
			}
			if (FinalPoint.getY() < initialPoint.getY()) {
				squarePosition.setY(initialPoint.getY() - square_length);
			}
			this.rectangle = new Rectangle(squarePosition, square_length, square_length, this.model.getIsFilled());
			model.addShape((Shape) this.rectangle);
			this.setFinalShape(false);
		}
			
		this.setChanged();
		this.notifyObservers();
	}
			 
	@Override
	public void setInitialPoint(Point point) {
		initialPoint = point;
	}
	@Override
	public void setDraggingPoint(Point point) {
		draggingPoint = point;
	}
	@Override
	public void setFinalPoint(Point point) {
		this.FinalPoint = point;
		
	}
	@Override
	public void setFinalShape(boolean bool) {
		this.FinalShape = bool;
		
	}

}