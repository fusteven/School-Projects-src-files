package ca.utoronto.utm.paint;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
/**
 * This is the top level View+Controller, it contains other aspects of the View+Controller.
 * @author arnold
 *
 */
public class View extends JFrame implements ActionListener {
	
	private static final long serialVersionUID = 1L;

	// The components that make this up
	private PaintPanel paintPanel;

	private ShapeChooserPanel shapeChooserPanel;
	private ColorSelectorPanel colorSelectorPanel;
	private JMenuBar menuPanel;
	public PaintModel model ;
	
	private String paintPanelImage; // Background image of PaintPanel



	private String image;
	
	/**
	 * This constructor takes in PaintModel and  necessary takes the PaintPanel, color selector panel, 
	 * shape chooser panel and  
	 * @param model
	 */
	public View(PaintModel model, String image) {
		super("Paint"); // Set the title and do other JFrame init
		this.model = model;
		Container c = this.getContentPane();
		this.setLayout(new BorderLayout());
	
		
		
		// Get the image to set as the BG for PaintModel
		
		BufferedImage bufferedImage = null;
		try {
			bufferedImage = ImageIO.read(new File(image));
		} catch (IOException e) {
			e.printStackTrace();
		}

		this.shapeChooserPanel = new ShapeChooserPanel(this);
		c.add(this.shapeChooserPanel, BorderLayout.WEST);
		
		this.paintPanel = new PaintPanel(this.model, this, bufferedImage);
		c.add(this.paintPanel, BorderLayout.CENTER);
		this.colorSelectorPanel = new ColorSelectorPanel(this);
		this.model.addObserver(this.colorSelectorPanel);
		c.add(this.colorSelectorPanel, BorderLayout.SOUTH);
		
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setJMenuBar(this.createMenuBar());
		this.pack();
		this.setSize(1280,850);
		this.setVisible(true);
		this.setResizable(false);
	}

	/**
	 * This private method draws the MenuBar on top of the paint program
	 * 
	 * @return JMenuBar
	 * @author csc207 student
	 */

	private JMenuBar createMenuBar() {
		JMenuBar menuBar = new JMenuBar();
		JMenu menu;
		JMenuItem menuItem;
		JRadioButtonMenuItem rbMenuItem;
		
		menuBar.setBorderPainted(false);
		menuBar.setCursor( new Cursor(Cursor.DEFAULT_CURSOR));
		
		menu = new JMenu("File"); // FILE Menu
		menu.setCursor( new Cursor(Cursor.HAND_CURSOR));

		// A group of JMenuItems
		menuItem = new JMenuItem("New");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuItem = new JMenuItem("Open");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuItem = new JMenuItem("Save");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menu.addSeparator(); // -------------

		menuItem = new JMenuItem("Exit");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuBar.add(menu);

		menu = new JMenu("Edit"); // EDIT Menu
		menu.setCursor( new Cursor(Cursor.HAND_CURSOR));

		// A group of JMenuItems
		menuItem = new JMenuItem("Cut");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuItem = new JMenuItem("Copy");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuItem = new JMenuItem("Paste");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menu.addSeparator(); // -------------

		menuItem = new JMenuItem("Undo");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuItem = new JMenuItem("Redo");
		menuItem.addActionListener(this);
		menuItem.setCursor( new Cursor(Cursor.HAND_CURSOR));
		menu.add(menuItem);

		menuBar.add(menu);
		
		
		menu = new JMenu("Background"); // CHANGE BG Menu
		menu.setCursor( new Cursor(Cursor.DEFAULT_CURSOR));
		ButtonGroup bgButtons = new ButtonGroup();
		
		rbMenuItem = new JRadioButtonMenuItem("Image 1");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);

		rbMenuItem = new JRadioButtonMenuItem("Image 2");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);
		
		rbMenuItem = new JRadioButtonMenuItem("Image 3");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);
		
		rbMenuItem = new JRadioButtonMenuItem("Image 4");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);
		
		rbMenuItem = new JRadioButtonMenuItem("Image 5");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);
		
		rbMenuItem = new JRadioButtonMenuItem("Image 6");
		rbMenuItem.addActionListener(this);
		bgButtons.add(rbMenuItem);
		menu.add(rbMenuItem);
		//this.selectIfImageButton(rbMenuItem);
		
		menuBar.add(menu);

		
		// Create fill style menu
		menu = new JMenu("Fill Style"); // FILL STYLE Menu
		menu.setCursor( new Cursor(Cursor.HAND_CURSOR));
		
		ButtonGroup fillButtons = new ButtonGroup();
		JRadioButtonMenuItem outline = new JRadioButtonMenuItem("Outline");
		outline.addActionListener(this);
		outline.setCursor( new Cursor(Cursor.HAND_CURSOR));
		fillButtons.add(outline);
		menu.add(outline);
		
		JRadioButtonMenuItem solid = new JRadioButtonMenuItem("Solid");
		solid.addActionListener(this);
		solid.setCursor( new Cursor(Cursor.HAND_CURSOR));
		fillButtons.add(solid);
		menu.add(solid);
		
		menuBar.add(menu);
		
		// Determine which fillButtons radio button to select on startup
		if (this.model.getIsFilled())
			solid.setSelected(true);
		else
			outline.setSelected(true);
	
		menu = new JMenu("Stroke"); // STROKE Menu
		menu.setCursor( new Cursor(Cursor.HAND_CURSOR));
		
		JMenu subMenu = new JMenu("Thickness"); // THICKNESS SubMenu
		subMenu.setCursor( new Cursor(Cursor.HAND_CURSOR));
		
		ThicknessSelectorPanel thicknessSelector = new ThicknessSelectorPanel(this);
		this.model.addObserver(thicknessSelector);
		subMenu.add(thicknessSelector);
		menu.add(subMenu);
		

		subMenu = new JMenu("Styles"); // STYLES SubMenu
		subMenu.setCursor( new Cursor(Cursor.HAND_CURSOR));
		ButtonGroup styleButtons = new ButtonGroup();
		
		JRadioButtonMenuItem solidStroke = new JRadioButtonMenuItem("Default");
		solidStroke.addActionListener(this);
		solidStroke.setCursor( new Cursor(Cursor.HAND_CURSOR));
		styleButtons.add(solidStroke);
		subMenu.add(solidStroke);
		
		JRadioButtonMenuItem dashedStroke = new JRadioButtonMenuItem("Dashed");
		dashedStroke.addActionListener(this);
		dashedStroke.setCursor( new Cursor(Cursor.HAND_CURSOR));
		styleButtons.add(dashedStroke);
		subMenu.add(dashedStroke);
		
		JRadioButtonMenuItem dottedStroke = new JRadioButtonMenuItem("Dotted Dashed");
		dottedStroke.addActionListener(this);
		dottedStroke.setCursor( new Cursor(Cursor.HAND_CURSOR));
		styleButtons.add(dottedStroke);
		subMenu.add(dottedStroke);

		solidStroke.setSelected(true);
		menu.add(subMenu);
		
		menuBar.add(menu);	
		

		return menuBar;
	}
	
	/**
	 * Helper function for the construction of Background menu.
	 * Sets button as the selected button if it corresponds with  this.paintPanelImage
	 * 
	 * @param button
	 * @return
	 */
	/*
	private void selectIfImageButton(JRadioButtonMenuItem button) {
		String buttonText = button.getText();
		// length - 1 puts length of string paintPanelImage in terms of indices
		// - 4 subracts ".jpg" or ".png" and moves one index to the left
		char imageNumber = paintPanelImage.charAt( paintPanelImage.length()-1 - 4);
		
		if (buttonText.equals("Image " + imageNumber))
			button.setSelected(true);
		else
			button.setSelected(false);
	}
	*/
	public PaintPanel getPaintPanel(){
		return paintPanel;
	}
	public ShapeChooserPanel getShapeChooserPanel() {
		return shapeChooserPanel;
	}
	
	public void actionPerformed(ActionEvent e) {
		if(e.getActionCommand()=="Solid")
			this.model.setSolid();
		else
			this.model.setOutline();
		
		if (e.getActionCommand()=="Default") {
			this.model.setStrokeType(model.DEFAULT);
		} else if (e.getActionCommand()=="Dashed") {
			this.model.setStrokeType(model.DASHED);
		} else if (e.getActionCommand()=="Dotted Dashed") {
			this.model.setStrokeType(model.DOTTED);
		}

		if (e.getActionCommand() == "Save") {
			JFrame parentFrame = new JFrame();
			shapeChooserPanel.setVisible(false);
			shapeChooserPanel.setBackground(Color.WHITE);
			BufferedImage image = new BufferedImage(paintPanel.getSize().width,
													paintPanel.getSize().height, 
													BufferedImage.TYPE_INT_ARGB); 
			Graphics g = image.createGraphics();
			this.paint(g);  
			g.dispose();

            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setCurrentDirectory(null);
            FileNameExtensionFilter pngFilter = new FileNameExtensionFilter("PNG file (*.png)", "png");
            fileChooser.addChoosableFileFilter(pngFilter);
            fileChooser.setFileFilter(pngFilter);

            int status = fileChooser.showSaveDialog(parentFrame);

            if (status == JFileChooser.APPROVE_OPTION) {
                try {
                    ImageIO.write(image, "png", fileChooser.getSelectedFile());
                    JOptionPane.showMessageDialog(null, "Image saved to "
                            + fileChooser.getSelectedFile().getName());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
            shapeChooserPanel.setVisible(true);
            shapeChooserPanel.setBackground(null);
		} else if (e.getActionCommand() == "Exit") {
			System.exit(0);
		}
			
		else if (e.getActionCommand() == "New"|| e.getActionCommand() == "Image 1"){
			new Paint("images/image1.jpg"); 
		} else if (e.getActionCommand() == "Image 2") {
			new Paint("images/image2.jpg");
		} else if (e.getActionCommand() == "Image 3") {
			new Paint("images/image3.jpg");	
		} else if (e.getActionCommand() == "Image 4") {
			new Paint("images/image4.jpg");	
		} else if (e.getActionCommand() == "Image 5") {
			new Paint("images/image5.png");	
		} else if (e.getActionCommand() == "Image 6") {
			new Paint("images/image6.jpg");	
		}
		
	}
}
