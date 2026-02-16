"""Request/response schemas for API validation."""

FLIGHT_PRICE_REQUEST = {
    "type": "object",
    "required": ["from", "to", "flightType", "agency", "distance", "time"],
    "properties": {
        "from": {"type": "string"},
        "to": {"type": "string"},
        "flightType": {"type": "string", "enum": ["economic", "premium", "firstClass"]},
        "agency": {"type": "string"},
        "distance": {"type": "number"},
        "time": {"type": "number"},
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
        "day_of_week": {"type": "integer", "minimum": 0, "maximum": 6}
    }
}

FLIGHT_CLASS_REQUEST = {
    "type": "object",
    "required": ["from", "to", "agency", "price", "distance", "time"],
    "properties": {
        "from": {"type": "string"},
        "to": {"type": "string"},
        "agency": {"type": "string"},
        "price": {"type": "number"},
        "distance": {"type": "number"},
        "time": {"type": "number"},
        "month": {"type": "integer"},
        "day_of_week": {"type": "integer"}
    }
}
