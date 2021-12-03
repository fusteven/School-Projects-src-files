package ca.utoronto.utm.paint;

import java.util.Observable;


/**
* This class handles all the necessary calculations, regardless, if
* the mouse is dragging or the mouse is released. 
* 
* @author csc207 Student
*
*/
public class TriangleStrategy extends Observable implements StrategyInterface{
	private Point initialPoint;
	private Point draggingPoint;
	private PaintModel model;
	private Triangle triangle;
	private Point FinalPoint;
	private boolean FinalShape = false;
	
	/**
	 * This constructor takes in the model and adds the shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public TriangleStrategy(PaintModel model) {
		this.model = model;
	}
	
	/**
	 * This method is called to do the necessary calculations with the initial, dragging
	 * and final mouse click positions for the triangle. 
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void behaviour() {
		if (initialPoint != null && draggingPoint != null && !FinalShape) {
			
			Point thirdVertex = new Point(this.draggingPoint.getX(), this.initialPoint.getY());
			model.setDraggedShape((Shape) new Triangle(this.initialPoint, this.draggingPoint, thirdVertex, this.model.getIsFilled()));
		}
		else if (FinalShape && FinalPoint != null) {
			Point thirdVertex = new Point(this.FinalPoint.getX(), this.initialPoint.getY());
			this.triangle = new Triangle(this.initialPoint, this.FinalPoint, thirdVertex, this.model.getIsFilled());
			this.model.addShape((Shape) this.triangle);
			this.setFinalShape(false);
		}
		
	}

	@Override
	public void setInitialPoint(Point point) {
		this.initialPoint = point;
		
	}

	@Override
	public void setDraggingPoint(Point point) {
		this.draggingPoint = point;
		
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
