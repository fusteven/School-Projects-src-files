package ca.utoronto.utm.paint;

/**
 * This class handles the Point that is created. It decides what the x and y values are,
 * the color of the points and the thickness. 
 *
 * @author csc207 student
 *
 */
public class Point {
	private int x, y;  // x and y co-ordinates of Point
	
	/**
	 * Create a Point at (0,0)
	 */
	public Point() {
		this(0, 0);
	}
	/**
	 * Create a Point at (x,y)
	 * 
	 * @param x : The x co-ordinate of Point
	 * @param y : THe y co-ordinate of Point
	 */
	public Point(int x, int y){
		this.x = x; this.y = y;
	}
	
	/**
	 * Return the x co-ordinate of Point
	 * 
	 * @return : int point to the x co-ordinate
	 */
	public int getX() {
		return x;
	}
	/**
	 * Set the x co-ordinate of Point
	 * 
	 * @param x : int describing the x co-ordinate of Point
	 */
	public void setX(int x) {
		this.x = x;
	}

	/**
	 * Return the y co-ordinate of Point
	 * 
	 * @return : int point to the y co-ordinate
	 */
	public int getY() {
		return y;
	}
	/**
	 * Set the y co-ordinate of Point
	 * 
	 * @param y : int describing the y co-ordinate of Point
	 */
	public void setY(int y) {
		this.y = y;
	}
	
}