package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.Observable;
/**
 * This class handles all the necessary calculations, regardless, if
 * the mouse is dragging or the mouse is released. 
 * 
 * @author csc207 Student
 *
 */
	
public class RectangleStrategy extends Observable implements StrategyInterface  {


	private Point initialPoint;
	private Point draggingPoint;
	private PaintModel model;
	private Rectangle rectangle;
	private Point FinalPoint;
	private boolean FinalShape = false;
	
	/**
	 * This constructor takes in the model and adds the shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public RectangleStrategy(PaintModel model) {
		this.model = model;
	}
	/**
	 * This method draws the actual rectangle. It utilizes the Graphics2D's built-in 
	 * methods.
	 * 
	 * @author csc207 Student
	 */
	public void behaviour() {
		if (initialPoint != null && draggingPoint != null) {
			Point default_initial = new Point(initialPoint.getX(), initialPoint.getY());
			int length_drag = Math.abs(draggingPoint.getX() - initialPoint.getX());
			int width_drag = Math.abs(draggingPoint.getY() - initialPoint.getY());

			if (draggingPoint.getX() < initialPoint.getX()) {
				default_initial.setX(draggingPoint.getX());
			}
			if (draggingPoint.getY() < initialPoint.getY()) {
				default_initial.setY(draggingPoint.getY());
			}
			model.setDraggedShape((Shape) new Rectangle(default_initial, width_drag, length_drag, this.model.getIsFilled()));

		}
			
		if (FinalShape != false && FinalPoint != null) {	
			Point rectPosition = new Point(initialPoint.getX(),initialPoint.getY());	
			if (FinalPoint.getX() < rectPosition.getX()) {
				rectPosition.setX(FinalPoint.getX());
			}	
			if (FinalPoint.getY() < initialPoint.getY()) {
				rectPosition.setY(FinalPoint.getY());
			}
			int length_final = Math.abs(FinalPoint.getX() - initialPoint.getX());
			int width_final = Math.abs(FinalPoint.getY() - initialPoint.getY()); 
			this.rectangle = new Rectangle(rectPosition, width_final, length_final, this.model.getIsFilled());
			this.model.addShape((Shape) this.rectangle);
			this.setFinalShape(false);
			
		this.setChanged();
		this.notifyObservers();
		}
		
			 
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
		
		FinalPoint = point;
		
	}
	@Override
	public void setFinalShape(boolean bool) {
		this.FinalShape = bool;
		
	}

}