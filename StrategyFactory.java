package ca.utoronto.utm.paint;

/**
 * This class will detect which shape button is pressed and create the necessary 
 * strategy
 * 
 * @author csc207 student
 *
 */
public class StrategyFactory {

	private static StrategyFactory instance = null;
	private PaintModel model;
	
	private StrategyFactory() {}
	
	public static synchronized StrategyFactory getInstance() {
		if (instance == null)
			instance = new StrategyFactory();
		return instance;
	}
	/**
	 * This method will return the necesaary strategies
	 * 
	 * @param id
	 * @return
	 */
	public StrategyInterface createStrategy(String id) {
		if (id.equalsIgnoreCase("Circle")) {
			return new CircleStrategy(this.model);
		} else if (id.equalsIgnoreCase("Rectangle")) {
			return new RectangleStrategy(this.model);
		} else if (id.equalsIgnoreCase("Square")) {
			return new SquareStrategy(this.model);
		} else if (id.equalsIgnoreCase("Squiggle")) {
			return new SquiggleStrategy(this.model);
		} else if (id.equalsIgnoreCase("Polyline")) {
			return new PolylineStrategy(this.model);

		} else if (id.equalsIgnoreCase("Triangle"))
			return new TriangleStrategy(this.model);
		return null;
	}
	
	public void setModel(PaintModel model) {
		this.model = model;
	}
	
}