package ca.utoronto.utm.paint;

import java.util.Observable;


/**
 * This class handles all the necessary calculations, regardless, if
 * the mouse is dragging or the mouse is released. 
 * 
 * @author csc207 Student
 *
 */
public class PolylineStrategy extends Observable implements StrategyInterface  {

	private PaintModel model;
	private Point initialPoint;
	private Point draggingPoint;
	private Point FinalPoint;
	
	private boolean FinalShape;
	private boolean rightClicked;

	/**
	 * This constructor takes in the model and adds the shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public PolylineStrategy(PaintModel model) {
		this.model = model;
	}
	/**
	 * This method is called to do the necessary calculations with the initial, dragging
	 * and final mouse click positions for polyline. 
	 * 
	 * @author csc207 Student
	 */
	public void behaviour() {
		if (initialPoint != null && FinalShape) {
			Polyline polyline = (Polyline) this.model.getDraggedShape();
			if (polyline.length() > 0) {
				if (rightClicked) {  // End current Polyline
					model.addShape(model.getDraggedShape());
					model.setDraggedShape(null);
					this.setRightClicked(false); // Reset right-click property
				} else {  // Else left-click: extend Polyline
					polyline.extend( new Point(draggingPoint.getX(), draggingPoint.getY()));
					model.setDraggedShape(polyline);
				}
				this.setChanged();
				this.notifyObservers();
			}
			
			this.setFinalShape(false);
		}
	}
	
	public void setInitialPoint(Point point) {
		initialPoint = point;
		if (!(this.model.getDraggedShape() instanceof Polyline)) {
			this.model.setDraggedShape( new Polyline(this.initialPoint));
			this.setChanged();
			this.notifyObservers();
		}
	}
	
	public void setDraggingPoint(Point point) {
		draggingPoint = point;
	}
	@Override
	public void setFinalPoint(Point point) {
		this.FinalPoint = point;
	}
	
	@Override
	public void setFinalShape(boolean bool) {
		this.FinalShape  = bool;
	}
	public void setRightClicked(boolean rightClicked) {
		this.rightClicked = rightClicked;
	}

}