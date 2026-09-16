"""Tool definitions and simulated execution environment for Chapter 3."""

from __future__ import annotations
from typing import Any, Dict


import json
import os

# Load real physical database from shared/datasets/data/ecommerce_db.json
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared", "datasets", "data", "ecommerce_db.json")

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        _db = json.load(f)
        ORDERS_DB = _db.get("orders", {})
        CUSTOMERS_DB = _db.get("customers", {})
else:
    ORDERS_DB = {}
    CUSTOMERS_DB = {}


def get_weather(zip_code: str) -> Dict[str, Any]:
    """Retrieve weather forecast for zip code."""
    if not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
        return {"status": "error", "error": f"Invalid zip code format: {zip_code}"}
    return {"status": "success", "zip_code": zip_code, "condition": "Sunny", "temp_f": 68}


def search_customer(customer_id: str = None, name: str = None) -> Dict[str, Any]:
    """Search for customer profile by ID or name."""
    if customer_id and customer_id in CUSTOMERS_DB:
        return {"status": "success", "customer": CUSTOMERS_DB[customer_id]}
    if name:
        for c in CUSTOMERS_DB.values():
            if name.lower() in c["name"].lower():
                return {"status": "success", "customer": c}
    return {"status": "error", "error": f"Customer not found for query (id={customer_id}, name={name})"}


def get_order(order_id: str) -> Dict[str, Any]:
    """Fetch order details from order management system."""
    if not order_id:
        return {"status": "error", "error": "Missing required argument: order_id"}
    if order_id in ORDERS_DB:
        return {"status": "success", "order": ORDERS_DB[order_id]}
    return {"status": "error", "error": f"Order #{order_id} not found in database"}


def calculate_refund(order_id: str) -> Dict[str, Any]:
    """Calculate refund eligibility based on purchase date."""
    if order_id not in ORDERS_DB:
        return {"status": "error", "error": f"Cannot calculate refund: Order #{order_id} does not exist"}
    order = ORDERS_DB[order_id]
    eligible = order["days_since_purchase"] <= 30
    refund_amount = order["amount"] if eligible else 0.0
    return {
        "status": "success",
        "order_id": order_id,
        "eligible": eligible,
        "refund_amount": refund_amount,
        "policy_rule": "30-day return policy",
    }


def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Simulate dispatching an email notification."""
    if not to or "@" not in to:
        return {"status": "error", "error": f"Invalid recipient email: {to}"}
    if not subject or not body:
        return {"status": "error", "error": "Email subject and body cannot be empty"}
    return {"status": "success", "dispatched_to": to, "message_id": f"msg-{abs(hash(to + subject)) % 10000}"}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_customer": search_customer,
    "get_order": get_order,
    "calculate_refund": calculate_refund,
    "send_email": send_email,
}
