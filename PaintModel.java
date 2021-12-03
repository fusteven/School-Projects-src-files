package ca.utoronto.utm.paint;

import java.util.ArrayList;
import java.util.Observable;
/**
 * It stores the shapes created from specific strategies and makes shape drawing commands
 * 
 * 
 * @author csc207 student
 *
 */
public class PaintModel extends Observable{

	private Shape draggedShape;
	private ArrayList<Shape> shapes;
	private ShapeDrawingCommand draggedShapeCommand;
	private boolean isFilled;
	private int strokeType;
	public static final int DEFAULT = 0;
	public static final int DASHED = 1;
	public static final int DOTTED = 2;

	/**
	 * This constructor initiates a list for all the shapes to be added in.
	 * It also initiates the current type of stroke
	 * 
	 * @author csc207 Student
	 */
	public PaintModel() {
		this.shapes = new ArrayList<Shape>();
		this.isFilled = false;
		this.strokeType = DEFAULT;
	}
	
	/**
	 * This method creates the drawing commands specifically for the dragged shape
	 * (When the cursor is dragging)
	 * 
	 * @author csc207 Student
	 * @param shape
	 */
	public void setDraggedShape (Shape shape) {
		this.draggedShape = shape;
		if (shape instanceof Circle) {
			draggedShapeCommand = new CircleDrawingCommand((Circle) draggedShape);
		}else if (shape instanceof Rectangle) {
			draggedShapeCommand = new RectangleDrawingCommand((Rectangle)draggedShape);
		}else if (shape instanceof Polyline) {
			draggedShapeCommand = new PolyLineDrawingCommand((Polyline)draggedShape);
		}else if (shape instanceof Squiggle) {
			draggedShapeCommand = new SquiggleDrawingCommand((Squiggle)draggedShape);
		}else if (shape instanceof Triangle) {
			draggedShapeCommand = new TriangleDrawingCommand((Triangle)draggedShape);
		}
		

		this.setChanged();
		this.notifyObservers("dragged shape");	
	}


	/**
	 * This method adds the shape to the ArrayList shapes
	 * 
	 * @author csc207 student
	 * 
	 */
	public void addShape(Shape shape) {
		this.shapes.add(shape);
		this.setChanged();
		this.notifyObservers("Shape added");
	}

	/**
	 * This method returns the ArrayList of shapes
	 *
	 *@author csc207 student
	 */
	public ArrayList<Shape> getShapes() {
		return shapes;
	}
	
	/**
	 * This method creates the drawing commands specifically for the final shape
	 * (When the cursor is released)
	 * 
	 * @author csc207 Student
	 * @param shapeDrawingList
	 */
	public void makeShapeCommands(ArrayList<ShapeDrawingCommand> shapeDrawingList) {
		for (Shape shape: this.shapes) {
			if (shape instanceof Circle) {
				CircleDrawingCommand circle = new CircleDrawingCommand((Circle)shape);
				shapeDrawingList.add(circle);
			}else if (shape instanceof Rectangle) {
				RectangleDrawingCommand rectangle = new RectangleDrawingCommand((Rectangle)shape);
				shapeDrawingList.add(rectangle);
			}else if (shape instanceof Polyline) {
				PolyLineDrawingCommand polyline = new PolyLineDrawingCommand((Polyline)shape);
				shapeDrawingList.add(polyline);
			}else if (shape instanceof Squiggle) {
				SquiggleDrawingCommand squiggle = new SquiggleDrawingCommand((Squiggle)shape);
				shapeDrawingList.add(squiggle);
			}else if (shape instanceof Triangle) {
				TriangleDrawingCommand triangle = new TriangleDrawingCommand((Triangle) shape);
				shapeDrawingList.add(triangle);
			}
		}
		this.setChanged();
		this.notifyObservers();
	}


	/**
	 * This method returns the shape that is being dragged
	 * 
	 * @return draggedShape
	 */
	public Shape getDraggedShape() {
		return draggedShape;

	}

	public ShapeDrawingCommand getDraggedShapeCommand() {
		return draggedShapeCommand;
	}
	
	public void setSolid() {
		this.isFilled = true;
	}
	
	public void setOutline() {
		this.isFilled = false;
	}
	
	public boolean getIsFilled() {
		return this.isFilled;
	}
	
	public int getStrokeType() {
		return this.strokeType;
	}
	
	public void setStrokeType(int stroke) {
		this.strokeType = stroke;
	}


}