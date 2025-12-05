"""
TourOptimizer API
Flask backend for tour route generation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import our tour router
sys.path.append(os.path.dirname(__file__))
from tour_router import TourRouter

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Initialize router with data
print("Loading tour data...")
DATA_PATH = os.path.join(os.path.dirname(__file__), 'us_only_with_states.csv')
router = TourRouter(DATA_PATH)
print(f"✓ Loaded {len(router.venues)} venues")

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is alive"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'venues_loaded': len(router.venues)
    })

# Main tour generation endpoint
@app.route('/api/generate-tour', methods=['POST'])
def generate_tour():
    """Generate optimized tour route"""
    try:
        # Get parameters from request
        data = request.json
        
        genre = data.get('genre', 'country')
        starting_city = data.get('startCity', 'Nashville')
        starting_state = data.get('startState', 'Tennessee')
        num_stops = int(data.get('numStops', 10))
        min_capacity = int(data.get('minCapacity', 0)) if data.get('minCapacity') else None
        max_capacity = int(data.get('maxCapacity', 999999)) if data.get('maxCapacity') else None
        optimize_for = data.get('optimizeFor', 'balanced')
        
        print(f"Generating {genre} tour from {starting_city}, {starting_state}")
        
        # Generate tour
        tour = router.build_tour_route(
            genre=genre,
            starting_city=starting_city,
            starting_state=starting_state,
            num_stops=num_stops,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            optimize_for=optimize_for
        )
        
        # Convert DataFrame to list of dicts
        tour_stops = []
        for idx, row in tour.iterrows():
            tour_stops.append({
                'venue': row['Venue'],
                'city': row['City'],
                'state': row['State'],
                'capacity': int(row['Capacity']),
                'revenue': float(row['Avg_Revenue']),
                'utilization': float(row['Avg_Capacity_Util']),
                'ticketPrice': float(row['Avg_Ticket_Price']),
                'events': int(row['Event_Count'])
            })
        
        # Calculate summary
        summary = {
            'totalRevenue': float(tour['Avg_Revenue'].sum()),
            'avgUtilization': float(tour['Avg_Capacity_Util'].mean()),
            'totalCapacity': int(tour['Capacity'].sum()),
            'statesCovered': int(tour['State'].nunique()),
            'avgTicketPrice': float(tour['Avg_Ticket_Price'].mean())
        }
        
        print(f"✓ Generated {len(tour_stops)} stop tour")
        
        return jsonify({
            'success': True,
            'stops': tour_stops,
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Get available genres
@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Return list of available genres"""
    genres = [
        'country', 'r&b', 'k-pop', 'rock', 'pop', 
        'hip-hop', 'metal', 'indie', 'electronic',
        'jazz', 'blues', 'reggae', 'folk'
    ]
    return jsonify({'genres': genres})

# Get market insights
@app.route('/api/market/<state>', methods=['GET'])
def get_market_insights(state):
    """Get insights for a specific state"""
    try:
        insights = router.get_market_insights(state)
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Download tour as CSV
@app.route('/api/download-tour', methods=['POST'])
def download_tour():
    """Download tour as CSV file"""
    try:
        data = request.json
        stops = data.get('stops', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(stops)
        
        # Save to CSV
        output_path = '/tmp/tour_route.csv'
        df.to_csv(output_path, index=False)
        
        return send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='tour_route.csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
