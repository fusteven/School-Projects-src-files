package ca.utoronto.utm.paint;

import java.awt.BasicStroke;
import java.awt.Graphics2D;

/**
 * This command will update the current stroke type
 * @author stevenfu
 *
 */
public class ThicknessSelectorCommand implements ShapeModifierCommand{

	private int thickness;
	private int strokeType;
	
	/**
	 * This constructor will take in the thickness and strokeID and udpate the drawing 
	 * conditions
	 * 
	 * @param thickness
	 * @param strokeID
	 * @author csc207 student
	 */
	public ThicknessSelectorCommand(int thickness, int strokeID) {
		this.thickness = thickness;
		this.strokeType = strokeID;
	}
	
	/**
	 * This method will update the drawing conditions
	 * 
	 * @author csc207 student
	 */
	@Override
	public void execute(Graphics2D g) {
		if(this.strokeType == PaintModel.DEFAULT)
			g.setStroke(new BasicStroke(this.thickness));
		else if (this.strokeType == PaintModel.DASHED) {
			g.setStroke(new BasicStroke(this.thickness, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{21,9}, 0));
		}	
		else 
			g.setStroke(new BasicStroke(this.thickness, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{21,9,4,5}, 0));
	}
}
