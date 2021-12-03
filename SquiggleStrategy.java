package ca.utoronto.utm.paint;


import java.util.Observable;

/**
* This class handles all the necessary calculations, regardless, if
* the mouse is dragging or the mouse is released. 
* 
* @author csc207 Student
*
*/
	
public class SquiggleStrategy extends Observable implements StrategyInterface  {


	private Point initialPoint;
	private Point draggingPoint;
	private PaintModel model;
	private boolean FinalShape = false;

	/**
	 * This constructor takes in the model and adds the shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public SquiggleStrategy(PaintModel model) {
		this.model = model;
	}
	
	/**
	 * This method is called to do the necessary calculations with the initial, dragging
	 * and final mouse click positions for squiggle. 
	 * 
	 * @author csc207 Student
	 */
	public void behaviour() {
		if (initialPoint != null && draggingPoint != null && !FinalShape) {
			if (initialPoint.getX()!= draggingPoint.getX() && initialPoint.getY()!= draggingPoint.getY()) {
				Squiggle currentSquiggle = (Squiggle) this.model.getDraggedShape();
				currentSquiggle.extend( new Point(draggingPoint.getX(), draggingPoint.getY()));
				this.model.setDraggedShape(currentSquiggle);
			} 
		}
		else if (initialPoint != null && draggingPoint != null && FinalShape) {
			Squiggle currentSquiggle = (Squiggle) model.getDraggedShape();
			currentSquiggle.extend( new Point(draggingPoint.getX(), draggingPoint.getY()));
			model.addShape(currentSquiggle);
			this.setFinalShape(false);
			this.setChanged();
			this.notifyObservers();
		}
	}
	public void setInitialPoint(Point point) {
		initialPoint = point;
		model.setDraggedShape((Shape) new Squiggle(initialPoint));
	}
	public void setDraggingPoint(Point point) {
		draggingPoint = point;
	}
	
	@Override
	public void setFinalPoint(Point point) {}
	
	@Override
	public void setFinalShape(boolean bool) {
		this.FinalShape  = bool;		
	}

}