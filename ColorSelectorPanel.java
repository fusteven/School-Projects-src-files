package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashMap;
import java.util.Observable;
import java.util.Observer;

import javax.swing.ImageIcon;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

  /**
   * This class creates a panel which allows the user to select pre-determined 
   * colours. 
   * 
   * @author csc207 Student
   *
   */
public class ColorSelectorPanel extends JPanel implements ActionListener, Observer {

	private View view;
	private HashMap colorMap;
	private RoundButton lastColor;
	private boolean first = true;
	private Color currentColor;
	
	/**
	 * This constructor takes in the view and adds the panel to it. It has a HashMap
	 * of color name and color. It also sets the font of the color names .
	 * 
	 * @author csc207 student 
	 * @param view a view object from the View class
	 * 
	 */
	public ColorSelectorPanel(View view) {
		
		HashMap<String, Color> colorMap = new HashMap<String, Color>();
		Color dgreen = new Color(0, 200, 100);
		Color[] colors = {Color.RED, Color.PINK, Color.ORANGE, Color.YELLOW, dgreen, Color.GREEN, Color.CYAN, Color.BLUE, Color.MAGENTA, Color.lightGray, Color.GRAY, Color.DARK_GRAY, Color.BLACK, Color.WHITE};
		String[] colorNames = {"Red", "Pink", "Orange", "Yellow", "DarkGreen", "Green", "Cyan", "Blue", "Magenta", "Light Grey", "Gray", "DarkGray", "Black", "White"};
		for (int i = 0; i<colors.length; i++) {
			colorMap.put(colorNames[i], colors[i]);
		}
		
		this.view = view;
		this.colorMap = colorMap;
		this.setBorder(new EmptyBorder(20, 25, -20, -25));
		this.setLayout(new GridLayout(1, 10));
		this.setCursor( new Cursor(Cursor.DEFAULT_CURSOR));
	
		Font font = new Font("Arial", Font.PLAIN, 0);
		for (int i = 0; i<colors.length; i++) {
			RoundButton button = new RoundButton(colorNames[i], colors[i]);
			this.add(button);
			button.setFont(font);
			button.addActionListener(this);
			button.setCursor( new Cursor(Cursor.HAND_CURSOR));
			if(colorNames[i] == "Black") {
				button.setIcon(new ImageIcon("images/tick.png"));
				this.lastColor = button;
			}
		}
	}

	/**
	 * This method does the action of setting the color button that is clicked 
	 * 
	 * @author csc207 student
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (this.first) {
			this.lastColor.setIcon(null);
		}
		RoundButton pressed = (RoundButton)e.getSource();
		pressed.setIcon(new ImageIcon("images/tick.png"));
		this.currentColor = (Color) this.colorMap.get(e.getActionCommand());;
		
		if (e.getSource() != lastColor) {
			this.lastColor.setIcon(null);
			this.lastColor = pressed;
		} 
	}

	/**
	 * This method receives the signal form the paint model to construct new color commands 
	 * for the new shape added to the paint model
	 * 
	 * @author csc207 Student
	 */
	@Override
	public void update(Observable arg0, Object arg1) {
		if (arg1 != null) {
			String command = (String) arg1;
			ColorSelectorCommand colorCommand = new ColorSelectorCommand(currentColor);
			if (command.equalsIgnoreCase("Shape added")) 
				this.view.getPaintPanel().addColorCommand(colorCommand);
			else 
				this.view.getPaintPanel().setDraggedColorCommand(colorCommand);
		}
	}

}

