package ca.utoronto.utm.paint;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTextField;

/**
 * This class creates a panel that allows the user to set the specific thickness of the line
 * 
 * @author csc207 student
 *
 */
public class ThicknessSelectorPanel extends JPanel implements ActionListener, Observer{

	private View view;
	private JTextField thickness;
	private int input;
	private ThicknessDisplayPanel display = new ThicknessDisplayPanel(this.input);
	private int currentThickness;

	/**
	 * This constructor takes in the View class and adds a panel where it asks for a int thickness
	 * 
	 * @author csc207 student
	 * @param view
	 */
	public ThicknessSelectorPanel(View view) {
		this.view = view;
		this.setLayout(new BorderLayout());
		this.setCursor( new Cursor(Cursor.DEFAULT_CURSOR));
		
		this.thickness = new JTextField(2);
		thickness.setText("1");
		thickness.addActionListener(this);
		
		JButton up = new JButton("up", new ImageIcon("images/up.png"));
		up.setBackground(new Color(225, 225, 225));
		up.setCursor( new Cursor(Cursor.HAND_CURSOR));
		up.addActionListener(this);
		
		JButton down = new JButton("down", new ImageIcon("images/down.png"));
		down.setBackground(new Color(225, 225, 225));
		down.setCursor( new Cursor(Cursor.HAND_CURSOR));
		down.addActionListener(this);
		
		this.add(display, BorderLayout.SOUTH);
		this.add(thickness, BorderLayout.WEST);
		this.add(up, BorderLayout.CENTER);
		this.add(down, BorderLayout.EAST);
	}

	/** 
	 * This method reacts to the inputs of the text field and the buttons.
	 * It restricts the thickness to be within 1-300.
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		try {
			this.currentThickness = Integer.parseInt(this.thickness.getText());
		} catch (NumberFormatException er) {
			System.err.println("Please enter integer for thickness.");
		}

		if (this.currentThickness < 1)
			this.currentThickness = 1;
		else if (this.currentThickness > 300)
			this.currentThickness = 300;
		else {
			if(e.getActionCommand()=="up")
				this.currentThickness++;
			else if(e.getActionCommand()=="down")
				this.currentThickness--;
		}

		this.thickness.setText(Integer.toString(this.currentThickness));
		this.display.setThickness(this.currentThickness);
		this.display.displayThickness();
	}				

	@Override
	public void update(Observable o, Object arg) {
		if (arg != null) {
			String command = (String) arg;
			ThicknessSelectorCommand thicknessCommand = new ThicknessSelectorCommand(this.currentThickness, this.view.model.getStrokeType());
			if (command.equalsIgnoreCase("Shape added")) 
				this.view.getPaintPanel().addThicknessCommand(thicknessCommand);
			else 
				this.view.getPaintPanel().setDraggedThicknessCommand(thicknessCommand);
		}

	}		
}
