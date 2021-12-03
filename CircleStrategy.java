package ca.utoronto.utm.paint;

/**
* This class handles all the necessary length and width calculations, regardless, if
* the mouse is dragging or the mouse is released. 
* 
* @author csc207 Student
*
*/
	
public class CircleStrategy implements StrategyInterface  {
	
	private Point initialPoint;
	private Point draggingPoint;
	private Point FinalPoint;
	private PaintModel model;
	private Circle circle;
	private boolean FinalShape = false;
	
	/**
	 * This constructor takes in the model and adds the shape that is created from either
	 * it is dragging or final shape.
	 * 
	 * @author csc207 Student
	 * @param model
	 */
	public CircleStrategy(PaintModel model) {
		this.model = model;
	}
	/**
	 * This method is called to do the necessary calculations with the initial, dragging
	 * and final mouse click positions for the length and width of the rectangle. 
	 * 
	 * @author csc207 Student
	 */
	public void behaviour( ) {
		
		if (initialPoint != null && !finalShape) {
			
			int circleRadius = (int) Math.hypot(Math.abs( draggingPoint.getX() - initialPoint.getX()), 
											Math.abs( draggingPoint.getY() - initialPoint.getY()));
			model.setDraggedShape((Shape) new Circle(initialPoint, circleRadius, this.model.getIsFilled()));
		}	
		

		if (FinalShape != false && FinalPoint != null) {

			circle = new Circle(initialPoint, 0, this.model.getIsFilled());
			double x = (double) FinalPoint.getX() - this.circle.getCentre().getX();
			double y = (double) FinalPoint.getY() - this.circle.getCentre().getY();
			int radius = (int)Math.hypot(x, y);
			
			this.circle.setRadius(radius);
			model.addShape(this.circle);
			this.setFinalShape(false);
		} 
	}
	
	@Override
	public void setInitialPoint(Point point) {
		this.initialPoint = point;
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