
import logging
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask Application Initialization ---
app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index() -> str:
    logger.info("Serving index page.")
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate() -> jsonify:
    logger.info("Handling calculation request.")

    if not request.is_json:
        logger.warning("Received non-JSON request for /calculate.")
        return jsonify({"success": False, "message": "Request must be JSON"}), 400

    data: Dict[str, Any] = request.get_json()
    num1 = data.get("num1")
    num2 = data.get("num2")
    operation = data.get("operation")

    if num1 is None or num2 is None or operation is None:
        logger.warning("Missing data in /calculate request.")
        return jsonify({"success": False, "message": "Missing 'num1', 'num2', or 'operation' in request data"}), 400

    try:
        num1_float = float(num1)
        num2_float = float(num2)

        logger.info(f"Received calculation request: {num1_float} {operation} {num2_float}")

        result: float = 0.0
        if operation == 'add':
            result = num1_float + num2_float
        elif operation == 'subtract':
            result = num1_float - num2_float
        elif operation == 'multiply':
            result = num1_float * num2_float
        elif operation == 'divide':
            if num2_float == 0:
                logger.error("Division by zero attempted.")
                return jsonify({"success": False, "message": "Division by zero is not allowed"}), 400
            result = num1_float / num2_float
        else:
            logger.warning(f"Unsupported operation received: {operation}")
            return jsonify({"success": False, "message": f"Unsupported operation: {operation}"}), 400

        logger.info(f"Calculation successful: {num1_float} {operation} {num2_float} = {result}")
        return jsonify({"success": True, "result": result})

    except ValueError:
        logger.error("Invalid number format received in /calculate request.")
        return jsonify({"success": False, "message": "Invalid number format for 'num1' or 'num2'"}), 400
    except Exception as e:
        logger.exception(f"An unexpected error occurred during calculation: {e}")
        return jsonify({"success": False, "message": f"An internal error occurred: {str(e)}"}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    logger.info("Starting Flask development server on http://127.0.0.1:5000/")
    app.run(debug=True, port=8080)
