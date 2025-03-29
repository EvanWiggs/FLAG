"""
User Interface Module

This module handles the user interface for the Food Order Verification System
using tkinter to create a simple GUI application.
"""

import os
import time
import threading
import logging
import queue
from typing import List, Set, Dict, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import cv2

# Import project modules
from food_verification.order_manager import OrderManager, Order, OrderItem
from food_verification.detector import FoodDetector, DetectionResult

logger = logging.getLogger(__name__)


class VideoFrame(ttk.Frame):
    """Frame for displaying video feed with detection overlays."""
    
    def __init__(self, parent, detector, order_manager):
        """
        Initialize the video frame.
        
        Args:
            parent: Parent widget
            detector: FoodDetector instance
            order_manager: OrderManager instance
        """
        super().__init__(parent, padding=5)
        
        self.detector = detector
        self.order_manager = order_manager
        
        # Initialize variables
        self.cap = None
        self.camera_index = 0
        self.is_running = False
        self.update_thread = None
        self.frame = None
        self.photo = None
        self.violations = set()
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        
        # Create a label for displaying the video
        self.video_label = ttk.Label(self)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Create a status label
        self.status_label = ttk.Label(
            self, 
            text="Camera not started",
            font=("Arial", 10),
            foreground="red"
        )
        self.status_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create control buttons
        self.start_button = ttk.Button(
            button_frame,
            text="Start Camera",
            command=self.start_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Camera",
            command=self.stop_camera,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.capture_button = ttk.Button(
            button_frame,
            text="Capture Frame",
            command=self.capture_frame,
            state=tk.DISABLED
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        # Create a frame counter and FPS display
        self.fps_label = ttk.Label(
            button_frame,
            text="FPS: 0",
            font=("Arial", 10)
        )
        self.fps_label.pack(side=tk.RIGHT, padx=5)
    
    def set_camera_index(self, index: int):
        """Set the camera index."""
        if self.is_running:
            self.stop_camera()
        
        self.camera_index = index
    
    def start_camera(self):
        """Start the camera and processing."""
        if self.is_running:
            return
        
        try:
            # Open the camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Start detector
            self.detector.start()
            
            # Update UI state
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.capture_button.config(state=tk.NORMAL)
            self.status_label.config(text="Camera running", foreground="green")
            
            # Start update thread
            self.update_thread = threading.Thread(
                target=self._update_thread,
                daemon=True
            )
            self.update_thread.start()
            
            logger.info(f"Started camera {self.camera_index}")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
            logger.error(f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop the camera and processing."""
        if not self.is_running:
            return
        
        try:
            # Stop detector
            self.detector.stop()
            
            # Release the camera
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Update UI state
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.capture_button.config(state=tk.DISABLED)
            self.status_label.config(text="Camera stopped", foreground="red")
            
            # Wait for thread to finish
            if self.update_thread:
                self.update_thread.join(timeout=2.0)
                self.update_thread = None
            
            logger.info("Stopped camera")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def capture_frame(self):
        """Save the current frame to a file."""
        if not self.is_running or self.frame is None:
            messagebox.showinfo("Capture", "No frame available to capture")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs("captures", exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captures/capture_{timestamp}.jpg"
            
            # Save frame
            cv2.imwrite(filename, self.frame)
            
            messagebox.showinfo("Capture", f"Frame captured and saved to {filename}")
            logger.info(f"Frame captured and saved to {filename}")
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to save frame: {e}")
            logger.error(f"Failed to save frame: {e}")
    
    def _update_thread(self):
        """Background thread for updating video frames."""
        logger.info("Video update thread started")
        
        last_fps_update = time.time()
        
        while self.is_running:
            try:
                # Grab a frame from the camera
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Store original frame
                self.frame = frame.copy()
                
                # Send frame for processing
                self.detector.process_frame(frame)
                
                # Get latest results
                result = self.detector.get_results()
                
                if result:
                    processed_frame, detections = result
                    
                    # Check for violations
                    active_item = self.order_manager.get_active_item()
                    if active_item:
                        violations, _ = self.detector.check_violations(detections, active_item)
                        self.violations = violations
                        
                        # Annotate the processed frame
                        annotated_frame = self.detector.annotate_frame(
                            processed_frame,
                            detections,
                            violations
                        )
                        
                        # Update the UI with the processed frame
                        self._update_image(annotated_frame)
                    else:
                        # Just show detections without checking violations
                        annotated_frame = self.detector.annotate_frame(
                            processed_frame,
                            detections
                        )
                        
                        # Update the UI with the processed frame
                        self._update_image(annotated_frame)
                else:
                    # No detection results yet, show the original frame
                    self._update_image(frame)
                
                # Update frame count and FPS
                self.frame_count += 1
                current_time = time.time()
                
                # Update FPS display every second
                if current_time - last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - last_fps_update)
                    self.frame_count = 0
                    last_fps_update = current_time
                    
                    # Update FPS label in UI thread
                    self.after(0, lambda: self.fps_label.config(text=f"FPS: {self.fps:.1f}"))
                
                # Sleep to limit update rate
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in video update thread: {e}")
                time.sleep(0.1)
        
        logger.info("Video update thread stopped")
    
    def _update_image(self, frame):
        """Update the image displayed in the UI."""
        try:
            # Convert the image from OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get the dimensions of the label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 1 and label_height > 1:
                # Resize the image to fit the label
                rgb_frame = cv2.resize(rgb_frame, (label_width, label_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update the label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference to prevent garbage collection
        except Exception as e:
            logger.error(f"Error updating image: {e}")


class OrderPanel(ttk.Frame):
    """Panel for displaying and managing orders."""
    
    def __init__(self, parent, order_manager):
        """
        Initialize the order panel.
        
        Args:
            parent: Parent widget
            order_manager: OrderManager instance
        """
        super().__init__(parent, padding=5)
        
        self.order_manager = order_manager
        
        # Create a frame for the order information
        order_frame = ttk.LabelFrame(self, text="Current Order")
        order_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Order ID and customer
        self.order_label = ttk.Label(
            order_frame,
            text="Order ID: N/A",
            font=("Arial", 12, "bold")
        )
        self.order_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.customer_label = ttk.Label(
            order_frame,
            text="Customer: N/A"
        )
        self.customer_label.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Separator(order_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        # Item information
        self.item_label = ttk.Label(
            order_frame,
            text="Current Item: N/A",
            font=("Arial", 11, "bold")
        )
        self.item_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Required ingredients
        required_frame = ttk.Frame(order_frame)
        required_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(
            required_frame,
            text="Required Ingredients:",
            width=20,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.required_label = ttk.Label(
            required_frame,
            text="N/A"
        )
        self.required_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Forbidden ingredients
        forbidden_frame = ttk.Frame(order_frame)
        forbidden_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(
            forbidden_frame,
            text="Forbidden Ingredients:",
            width=20,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.forbidden_label = ttk.Label(
            forbidden_frame,
            text="N/A"
        )
        self.forbidden_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Navigation buttons
        nav_frame = ttk.Frame(order_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.prev_button = ttk.Button(
            nav_frame,
            text="Previous Item",
            command=self.previous_item
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            nav_frame,
            text="Next Item",
            command=self.next_item
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self, text="Verification Status")
        status_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 11)
        )
        self.status_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Violations display
        self.violations_text = tk.Text(
            status_frame,
            height=5,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.violations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        """Update the display with current order information."""
        order = self.order_manager.get_active_order()
        item = self.order_manager.get_active_item()
        
        if order:
            self.order_label.config(text=f"Order ID: {order.order_id}")
            self.customer_label.config(text=f"Customer: {order.customer_name}")
        else:
            self.order_label.config(text="Order ID: N/A")
            self.customer_label.config(text="Customer: N/A")
        
        if item:
            self.item_label.config(text=f"Current Item: {item.name}")
            self.required_label.config(text=", ".join(item.required_ingredients) or "None")
            self.forbidden_label.config(text=", ".join(item.forbidden_ingredients) or "None")
        else:
            self.item_label.config(text="Current Item: N/A")
            self.required_label.config(text="N/A")
            self.forbidden_label.config(text="N/A")
    
    def next_item(self):
        """Switch to the next item in the order."""
        self.order_manager.next_item()
        self.update_display()
    
    def previous_item(self):
        """Switch to the previous item in the order."""
        self.order_manager.previous_item()
        self.update_display()
    
    def update_violations(self, violations: Set[str]):
        """
        Update the violations display.
        
        Args:
            violations: Set of ingredient names that violate the order
        """
        # Enable text widget for editing
        self.violations_text.config(state=tk.NORMAL)
        
        # Clear current text
        self.violations_text.delete(1.0, tk.END)
        
        if violations:
            self.violations_text.insert(tk.END, "VIOLATIONS DETECTED!\n\n")
            
            for violation in violations:
                self.violations_text.insert(
                    tk.END,
                    f"• '{violation}' should not be present\n"
                )
            
            # Highlight the text in red
            self.violations_text.tag_add("violations", 1.0, tk.END)
            self.violations_text.tag_config("violations", foreground="red")
            
            self.status_label.config(
                text="⚠️ Violations detected",
                foreground="red"
            )
        else:
            self.violations_text.insert(tk.END, "No violations detected.")
            self.status_label.config(
                text="✓ No violations",
                foreground="green"
            )
        
        # Disable text widget to prevent editing
        self.violations_text.config(state=tk.DISABLED)


class Application:
    """Main application class for the Food Order Verification System."""
    
    def __init__(self, 
                 order_manager: OrderManager,
                 detector: FoodDetector,
                 camera_index: int = 0,
                 fullscreen: bool = False):
        """
        Initialize the application.
        
        Args:
            order_manager: OrderManager instance
            detector: FoodDetector instance
            camera_index: Index of the camera to use
            fullscreen: Whether to start in fullscreen mode
        """
        self.order_manager = order_manager
        self.detector = detector
        self.camera_index = camera_index
        self.fullscreen = fullscreen
        
        # Create the root window
        self.root = tk.Tk()
        self.root.title("Food Order Verification System")
        self.root.geometry("1024x768")
        self.root.minsize(800, 600)
        
        if fullscreen:
            self.root.attributes("-fullscreen", True)
        
        # Configure the style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Use a modern looking theme
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Create a paned window for the main layout
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(expand=True, fill=tk.BOTH)
        
        # Create the video frame
        self.video_frame = VideoFrame(
            self.paned_window,
            detector=self.detector,
            order_manager=self.order_manager
        )
        self.paned_window.add(self.video_frame, weight=3)
        
        # Create the order panel
        self.order_panel = OrderPanel(
            self.paned_window,
            order_manager=self.order_manager
        )
        self.paned_window.add(self.order_panel, weight=1)
        
        # Create the menu bar
        self.create_menu()
        
        # Set up event handling
        self.setup_events()
        
        # Set the camera index
        self.video_frame.set_camera_index(camera_index)
    
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Start Camera", command=self.video_frame.start_camera)
        file_menu.add_command(label="Stop Camera", command=self.video_frame.stop_camera)
        file_menu.add_separator()
        file_menu.add_command(label="Capture Frame", command=self.video_frame.capture_frame)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_application)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Orders menu
        orders_menu = tk.Menu(menubar, tearoff=0)
        orders_menu.add_command(label="Load Orders", command=self.load_orders)
        orders_menu.add_command(label="Save Orders", command=self.save_orders)
        orders_menu.add_separator()
        orders_menu.add_command(label="Previous Item", command=self.order_panel.previous_item)
        orders_menu.add_command(label="Next Item", command=self.order_panel.next_item)
        menubar.add_cascade(label="Orders", menu=orders_menu)
        
        # Camera menu
        camera_menu = tk.Menu(menubar, tearoff=0)
        camera_menu.add_command(label="Select Camera", command=self.select_camera)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def setup_events(self):
        """Set up event handling."""
        # Set up periodic check for violations
        self.root.after(500, self.check_violations)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
    
    def check_violations(self):
        """Periodically check for violations and update the UI."""
        # Update the order panel with the latest violations
        self.order_panel.update_violations(self.video_frame.violations)
        
        # Schedule next check
        self.root.after(500, self.check_violations)
    
    def select_camera(self):
        """Open a dialog to select a camera."""
        camera_index = simpledialog.askinteger(
            "Select Camera",
            "Enter camera index (0, 1, 2, etc.):",
            initialvalue=self.camera_index,
            minvalue=0,
            maxvalue=10
        )
        
        if camera_index is not None:
            self.camera_index = camera_index
            self.video_frame.set_camera_index(camera_index)
    
    def load_orders(self):
        """Load orders from a file."""
        filename = filedialog.askopenfilename(
            title="Load Orders",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                # Update the orders file and load
                self.order_manager.orders_file = filename
                self.order_manager.load_orders()
                
                # Update the display
                self.order_panel.update_display()
                
                messagebox.showinfo(
                    "Load Orders",
                    f"Successfully loaded {len(self.order_manager.orders)} orders."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load orders: {e}")
    
    def save_orders(self):
        """Save orders to a file."""
        filename = filedialog.asksaveasfilename(
            title="Save Orders",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                # Update the orders file and save
                self.order_manager.orders_file = filename
                self.order_manager.save_orders()
                
                messagebox.showinfo(
                    "Save Orders",
                    f"Successfully saved {len(self.order_manager.orders)} orders."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save orders: {e}")
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About",
            "Food Order Verification System\n\n"
            "This application uses computer vision to verify that food orders "
            "are being prepared correctly, by detecting ingredients and comparing "
            "them against order specifications."
        )
    
    def exit_application(self):
        """Exit the application cleanly."""
        try:
            # Stop the camera if running
            self.video_frame.stop_camera()
            
            # Save orders
            if self.order_manager.orders_file:
                self.order_manager.save_orders()
            
            # Destroy the main window
            self.root.destroy()
            
            logger.info("Application exited normally")
        except Exception as e:
            logger.error(f"Error during exit: {e}")
    
    def run(self):
        """Run the application."""
        # Start the camera
        self.video_frame.start_camera()
        
        # Start the main event loop
        self.root.mainloop()