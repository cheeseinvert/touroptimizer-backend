"""
TourOptimizer API - With BULLETPROOF CORS
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)

# Let flask-cors handle ALL CORS
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False,
         "max_age": 3600
     }})

# Mock tour data
MOCK_TOURS = {
    'country': [
        {'venue': 'Nissan Stadium', 'city': 'Nashville', 'state': 'TN', 'capacity': 69143, 'revenue': 7287732, 'utilization': 97.6},
        {'venue': 'Kyle Field', 'city': 'College Station', 'state': 'TX', 'capacity': 110905, 'revenue': 25691143, 'utilization': 100.0},
        {'venue': 'Soldier Field', 'city': 'Chicago', 'state': 'IL', 'capacity': 49943, 'revenue': 17107701, 'utilization': 99.9},
        {'venue': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'capacity': 107256, 'revenue': 17199470, 'utilization': 100.0},
        {'venue': 'Gillette Stadium', 'city': 'Foxborough', 'state': 'MA', 'capacity': 112462, 'revenue': 14135021, 'utilization': 99.7},
        {'venue': 'Paycor Stadium', 'city': 'Cincinnati', 'state': 'OH', 'capacity': 107226, 'revenue': 14661999, 'utilization': 100.0},
        {'venue': 'Mercedes-Benz Stadium', 'city': 'Atlanta', 'state': 'GA', 'capacity': 49695, 'revenue': 15867905, 'utilization': 100.0},
        {'venue': 'Raymond James Stadium', 'city': 'Tampa', 'state': 'FL', 'capacity': 132787, 'revenue': 12593174, 'utilization': 100.0},
        {'venue': 'Lincoln Financial Field', 'city': 'Philadelphia', 'state': 'PA', 'capacity': 103162, 'revenue': 13078166, 'utilization': 100.0},
        {'venue': 'Bank Of America Stadium', 'city': 'Charlotte', 'state': 'NC', 'capacity': 50664, 'revenue': 11854132, 'utilization': 100.0},
    ],
    'r&b': [
        {'venue': 'Intuit Dome', 'city': 'Inglewood', 'state': 'CA', 'capacity': 11200, 'revenue': 2893143, 'utilization': 98.5},
        {'venue': 'Climate Pledge Arena', 'city': 'Seattle', 'state': 'WA', 'capacity': 15000, 'revenue': 2234567, 'utilization': 96.2},
        {'venue': 'United Center', 'city': 'Chicago', 'state': 'IL', 'capacity': 13456, 'revenue': 3456789, 'utilization': 99.1},
        {'venue': 'Madison Square Garden', 'city': 'New York', 'state': 'NY', 'capacity': 13000, 'revenue': 4567890, 'utilization': 100.0},
        {'venue': 'TD Garden', 'city': 'Boston', 'state': 'MA', 'capacity': 13000, 'revenue': 2345678, 'utilization': 97.8},
        {'venue': 'Prudential Center', 'city': 'Newark', 'state': 'NJ', 'capacity': 12000, 'revenue': 1987654, 'utilization': 95.4},
        {'venue': 'Spectrum Center', 'city': 'Charlotte', 'state': 'NC', 'capacity': 14000, 'revenue': 2876543, 'utilization': 98.0},
        {'venue': 'Little Caesars Arena', 'city': 'Detroit', 'state': 'MI', 'capacity': 14500, 'revenue': 2654321, 'utilization': 96.7},
        {'venue': 'Nationwide Arena', 'city': 'Columbus', 'state': 'OH', 'capacity': 14000, 'revenue': 2123456, 'utilization': 94.3},
        {'venue': 'Wells Fargo Center', 'city': 'Philadelphia', 'state': 'PA', 'capacity': 13500, 'revenue': 3234567, 'utilization': 99.2},
    ],
    'k-pop': [
        {'venue': 'Citi Field', 'city': 'New York', 'state': 'NY', 'capacity': 63000, 'revenue': 8765432, 'utilization': 100.0},
        {'venue': 'Wrigley Field', 'city': 'Chicago', 'state': 'IL', 'capacity': 38000, 'revenue': 5432109, 'utilization': 99.5},
        {'venue': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA', 'capacity': 95000, 'revenue': 12345678, 'utilization': 100.0},
        {'venue': 'Allegiant Stadium', 'city': 'Las Vegas', 'state': 'NV', 'capacity': 90000, 'revenue': 11234567, 'utilization': 100.0},
        {'venue': 'Capital One Arena', 'city': 'Washington', 'state': 'DC', 'capacity': 13000, 'revenue': 2987654, 'utilization': 98.7},
        {'venue': 'Footprint Center', 'city': 'Phoenix', 'state': 'AZ', 'capacity': 23000, 'revenue': 4321098, 'utilization': 97.3},
        {'venue': 'Oracle Park', 'city': 'San Francisco', 'state': 'CA', 'capacity': 42000, 'revenue': 6543210, 'utilization': 99.8},
        {'venue': 'T-Mobile Arena', 'city': 'Las Vegas', 'state': 'NV', 'capacity': 18000, 'revenue': 3654321, 'utilization': 98.9},
        {'venue': 'Toyota Center', 'city': 'Houston', 'state': 'TX', 'capacity': 15000, 'revenue': 2876543, 'utilization': 96.5},
        {'venue': 'American Airlines Center', 'city': 'Dallas', 'state': 'TX', 'capacity': 14000, 'revenue': 2543210, 'utilization': 95.8},
    ],
    'rock': [
        {'venue': 'Red Rocks Amphitheatre', 'city': 'Morrison', 'state': 'CO', 'capacity': 9525, 'revenue': 1543210, 'utilization': 100.0},
        {'venue': 'Greek Theatre', 'city': 'Los Angeles', 'state': 'CA', 'capacity': 5870, 'revenue': 987654, 'utilization': 99.2},
        {'venue': 'Hollywood Bowl', 'city': 'Los Angeles', 'state': 'CA', 'capacity': 17500, 'revenue': 2876543, 'utilization': 98.7},
        {'venue': 'Madison Square Garden', 'city': 'New York', 'state': 'NY', 'capacity': 13000, 'revenue': 3456789, 'utilization': 100.0},
        {'venue': 'The Fillmore', 'city': 'San Francisco', 'state': 'CA', 'capacity': 1315, 'revenue': 234567, 'utilization': 100.0},
    ],
    'pop': [
        {'venue': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'capacity': 82500, 'revenue': 14567890, 'utilization': 100.0},
        {'venue': 'SoFi Stadium', 'city': 'Inglewood', 'state': 'CA', 'capacity': 95000, 'revenue': 16789012, 'utilization': 100.0},
        {'venue': 'AT&T Stadium', 'city': 'Arlington', 'state': 'TX', 'capacity': 80000, 'revenue': 13456789, 'utilization': 99.8},
        {'venue': 'Soldier Field', 'city': 'Chicago', 'state': 'IL', 'capacity': 61500, 'revenue': 10234567, 'utilization': 99.5},
    ],
}

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'TourOptimizer API',
        'version': '1.0.0',
        'cors': 'enabled',
        'endpoints': {
            'health': '/api/health',
            'generate_tour': '/api/generate-tour (POST)',
            'genres': '/api/genres'
        }
    })

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Check if API is alive"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'venues_loaded': sum(len(tours) for tours in MOCK_TOURS.values()),
        'cors': 'enabled',
        'message': 'TourOptimizer API is running!'
    })

@app.route('/api/generate-tour', methods=['POST', 'OPTIONS'])
def generate_tour():
    """Generate optimized tour route"""
    # Handle OPTIONS preflight
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        data = request.json or {}
        
        genre = data.get('genre', 'country').lower()
        num_stops = int(data.get('numStops', 10))
        
        # Get tour data for genre (fallback to country if genre not found)
        tour_data = MOCK_TOURS.get(genre, MOCK_TOURS['country'])
        
        # Limit to requested number of stops
        stops = tour_data[:num_stops]
        
        # Calculate summary
        total_revenue = sum(stop['revenue'] for stop in stops)
        avg_utilization = sum(stop['utilization'] for stop in stops) / len(stops) if stops else 0
        total_capacity = sum(stop['capacity'] for stop in stops)
        states_covered = len(set(stop['state'] for stop in stops))
        
        summary = {
            'totalRevenue': total_revenue,
            'avgUtilization': round(avg_utilization, 1),
            'totalCapacity': total_capacity,
            'statesCovered': states_covered
        }
        
        return jsonify({
            'success': True,
            'stops': stops,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Return list of available genres"""
    return jsonify({
        'genres': list(MOCK_TOURS.keys())
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
