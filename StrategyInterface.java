package ca.utoronto.utm.paint;


/**
 * This interface will make all other strategies to have the required methods
 * @author stevenfu
 *
 */
public interface StrategyInterface  {
	
	boolean finalShape = false;

	 void behaviour();
	 void setInitialPoint(Point point);
	 void setDraggingPoint(Point point);
	 void setFinalPoint(Point point);
	 void setFinalShape(boolean bool);
}