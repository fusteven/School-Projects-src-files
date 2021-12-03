package ca.utoronto.utm.paint;

/**
 * This class allows the user to play an instance of the paint program. As soon as the program
 * runs, it will allow the user to do several features such as choosing the shape size or selecting
 * the shape color to draw. 
 * This class will be the app. It will create a Paint instance and create an object of 
 * PaintModel and move it into the View class. 
 */
public class Paint {
	private PaintModel model; // Model
	private View view; // View + Controller
	
	public static void main(String[] args) {
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				new Paint("images/image1.jpg"); // Set default BG image as image1.jpg
			}
		});
	}
	
	
	/**
	 * This construct creates a PaintModel and sends the object cerated to View.
	 * 
	 * @param image : The filename of the default image to use for the BG in View
	 */
	public Paint(String image) {
		// Create MVC components and hook them together
		this.model = new PaintModel(); // Model
		this.view = new View(this.model, image); // View+Controller
		StrategyFactory.getInstance().setModel(this.model);
	}
	
	/**
	 * This method returns the model object created in the constructor. 
	 * 
	 * @author csc207 student
	 * @return
	 */
	public PaintModel getModel() {
		return model;
	}
	/**
	 *  This method returns the view object created in the constructor. 
	 * 
	 * @return
	 */
	public View getView() {
		return view;
	}

}
