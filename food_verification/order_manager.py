"""
Order Manager Module

This module handles the management of food orders
and provides functionality to check prepared food
against the order specifications.
"""

import os
import json
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderItem:
    """Represents a single item in an order."""
    
    def __init__(self, 
                 item_id: str,
                 name: str, 
                 required_ingredients: List[str], 
                 forbidden_ingredients: List[str]):
        """
        Initialize an order item.
        
        Args:
            item_id: Unique identifier for the item
            name: Display name of the item
            required_ingredients: List of ingredients that must be included
            forbidden_ingredients: List of ingredients that must not be included
        """
        self.item_id = item_id
        self.name = name
        self.required_ingredients = set(required_ingredients)
        self.forbidden_ingredients = set(forbidden_ingredients)
    
    def __repr__(self):
        return f"OrderItem({self.name}, required={self.required_ingredients}, forbidden={self.forbidden_ingredients})"
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            'item_id': self.item_id,
            'name': self.name,
            'required_ingredients': list(self.required_ingredients),
            'forbidden_ingredients': list(self.forbidden_ingredients)
        }


class Order:
    """Represents a complete food order."""
    
    def __init__(self, 
                 order_id: str, 
                 customer_name: str = "", 
                 items: Optional[List[OrderItem]] = None):
        """
        Initialize an order.
        
        Args:
            order_id: Unique identifier for the order
            customer_name: Name of the customer
            items: List of OrderItem objects
        """
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items or []
        self.timestamp = datetime.now()
        self.status = "new"  # new, in_progress, completed, cancelled
    
    def __repr__(self):
        return f"Order({self.order_id}, items={len(self.items)})"
    
    def add_item(self, item: OrderItem):
        """Add an item to the order."""
        self.items.append(item)
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            'order_id': self.order_id,
            'customer_name': self.customer_name,
            'items': [item.to_dict() for item in self.items],
            'timestamp': self.timestamp.isoformat(),
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create an Order from dictionary data."""
        order = cls(
            order_id=data['order_id'],
            customer_name=data.get('customer_name', "")
        )
        
        # Add items
        for item_data in data.get('items', []):
            order.add_item(OrderItem(
                item_id=item_data.get('item_id', str(id(item_data))),
                name=item_data['name'],
                required_ingredients=item_data.get('required_ingredients', []),
                forbidden_ingredients=item_data.get('forbidden_ingredients', [])
            ))
        
        # Set other attributes
        if 'timestamp' in data:
            try:
                order.timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                order.timestamp = datetime.now()
        
        order.status = data.get('status', 'new')
        
        return order


class OrderManager:
    """
    Manages food orders and provides verification functionality.
    """
    
    def __init__(self, orders_file: str = None):
        """
        Initialize the order manager.
        
        Args:
            orders_file: Path to JSON file containing orders
        """
        self.orders_file = orders_file
        self.orders = {}
        self.active_order_id = None
        self.active_item_index = 0
        
        # Load sample orders if file doesn't exist
        if orders_file and os.path.exists(orders_file):
            self.load_orders()
        else:
            self._create_sample_orders()
    
    def _create_sample_orders(self):
        """Create sample orders for demonstration."""
        # Create a chicken sandwich order
        sandwich_order = Order("ORD-1001", "John Doe")
        
        # Add items to the order
        sandwich_order.add_item(OrderItem(
            item_id="ITEM-1",
            name="Chicken Sandwich",
            required_ingredients=["bread", "chicken", "lettuce"],
            forbidden_ingredients=["pickle", "onion"]
        ))
        
        sandwich_order.add_item(OrderItem(
            item_id="ITEM-2",
            name="French Fries",
            required_ingredients=["potato"],
            forbidden_ingredients=["salt"]
        ))
        
        # Save the order
        self.orders[sandwich_order.order_id] = sandwich_order
        
        # Create a burger order
        burger_order = Order("ORD-1002", "Jane Smith")
        
        # Add items to the order
        burger_order.add_item(OrderItem(
            item_id="ITEM-3",
            name="Beef Burger",
            required_ingredients=["bread", "beef", "lettuce", "tomato"],
            forbidden_ingredients=["cheese", "pickle"]
        ))
        
        # Save the order
        self.orders[burger_order.order_id] = burger_order
        
        # Set active order
        self.active_order_id = sandwich_order.order_id
    
    def save_orders(self):
        """Save orders to file."""
        if not self.orders_file:
            logger.warning("No orders file specified, orders will not be saved")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.orders_file), exist_ok=True)
            
            # Convert orders to dictionaries
            orders_dict = {
                order_id: order.to_dict()
                for order_id, order in self.orders.items()
            }
            
            # Save to JSON file
            with open(self.orders_file, 'w') as f:
                json.dump(orders_dict, f, indent=2)
            
            logger.info(f"Saved {len(self.orders)} orders to {self.orders_file}")
        except Exception as e:
            logger.error(f"Failed to save orders: {e}")
    
    def load_orders(self):
        """Load orders from file."""
        if not self.orders_file or not os.path.exists(self.orders_file):
            logger.warning(f"Orders file not found: {self.orders_file}")
            return
        
        try:
            # Load from JSON file
            with open(self.orders_file, 'r') as f:
                orders_dict = json.load(f)
            
            # Convert to Order objects
            self.orders = {
                order_id: Order.from_dict(order_data)
                for order_id, order_data in orders_dict.items()
            }
            
            # Set active order to the first one if any exist
            if self.orders:
                self.active_order_id = next(iter(self.orders))
            
            logger.info(f"Loaded {len(self.orders)} orders from {self.orders_file}")
        except Exception as e:
            logger.error(f"Failed to load orders: {e}")
    
    def add_order(self, order: Order):
        """
        Add a new order.
        
        Args:
            order: Order object to add
        """
        self.orders[order.order_id] = order
        
        # Set as active if no active order
        if not self.active_order_id:
            self.active_order_id = order.order_id
        
        # Save to file
        self.save_orders()
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: ID of the order to retrieve
            
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_active_order(self) -> Optional[Order]:
        """
        Get the currently active order.
        
        Returns:
            Active Order object or None if no active order
        """
        if not self.active_order_id:
            return None
        
        return self.orders.get(self.active_order_id)
    
    def get_active_item(self) -> Optional[OrderItem]:
        """
        Get the currently active item from the active order.
        
        Returns:
            Active OrderItem or None if no active order or item
        """
        order = self.get_active_order()
        if not order or not order.items:
            return None
        
        # Ensure index is valid
        if self.active_item_index >= len(order.items):
            self.active_item_index = 0
        
        return order.items[self.active_item_index]
    
    def set_active_order(self, order_id: str) -> bool:
        """
        Set the active order.
        
        Args:
            order_id: ID of the order to set as active
            
        Returns:
            True if successful, False if order not found
        """
        if order_id in self.orders:
            self.active_order_id = order_id
            self.active_item_index = 0
            return True
        
        return False
    
    def next_item(self) -> Optional[OrderItem]:
        """
        Move to the next item in the active order.
        
        Returns:
            The next OrderItem or None if no active order or no more items
        """
        order = self.get_active_order()
        if not order or not order.items:
            return None
        
        # Move to next item
        self.active_item_index = (self.active_item_index + 1) % len(order.items)
        
        return self.get_active_item()
    
    def previous_item(self) -> Optional[OrderItem]:
        """
        Move to the previous item in the active order.
        
        Returns:
            The previous OrderItem or None if no active order or no more items
        """
        order = self.get_active_order()
        if not order or not order.items:
            return None
        
        # Move to previous item
        self.active_item_index = (self.active_item_index - 1) % len(order.items)
        
        return self.get_active_item()
    
    def update_order_status(self, order_id: str, status: str) -> bool:
        """
        Update the status of an order.
        
        Args:
            order_id: ID of the order to update
            status: New status ('new', 'in_progress', 'completed', 'cancelled')
            
        Returns:
            True if successful, False if order not found
        """
        if order_id not in self.orders:
            return False
        
        valid_statuses = ['new', 'in_progress', 'completed', 'cancelled']
        if status not in valid_statuses:
            logger.warning(f"Invalid status: {status}. Must be one of {valid_statuses}")
            return False
        
        self.orders[order_id].status = status
        self.save_orders()
        
        return True