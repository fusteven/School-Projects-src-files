package ca.utoronto.utm.paint;

/**
 * This class is the parent class of all shape classes. 
 * 
 * @author csc207 students
 *
 */
public class Shape {
	
	private boolean isFilled;
	
	public Shape(boolean fill) {
		this.isFilled = fill;
	}
	
	public boolean getFilled() {
		return this.isFilled;
	}
}