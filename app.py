from flask import Flask, render_template, jsonify, request, send_file
import folium
import math
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import threading
import queue
import json
from werkzeug.utils import secure_filename
from models.traffic_model import TrafficSimulation, VehicleType, SIGNALS
from congestion.traffic_data import generate_traffic_data, get_traffic_flow
from congestion.model import train_models, predict_and_optimize
from sounddetection.detection import load_model, generate_spectrogram, classify_siren
import tensorflow as tf

# Add parent directory to Python path to find sounddetection module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from optimization_copy.ABC import optimizeSignal
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure all required packages are installed and the project structure is correct.")
    sys.exit(1)

app = Flask(__name__)

# Initialize models and simulation
try:
    simulation = TrafficSimulation(num_vehicles=20)
    
    # Track blocked roads and their durations
    blocked_roads = {}  # Format: {road_id: {'until': datetime, 'segment': (start, end)}}
    
    # Track simulation statistics
    simulation_stats = {
        'start_time': None,
        'total_vehicles': 0,
        'emergency_vehicles': 0,
        'avg_wait_time': 0,
        'system_efficiency': 0,
        'total_waiting_vehicles': 0,
        'average_wait_time': 0,
        'total_throughput': 0
    }

    # Initialize sound detection queue
    sound_detection_queue = queue.Queue()
    sound_detection_results = []
    
    # Initialize optimization results
    optimization_results = {}
    
    # Initialize ML models
    gnn_model, rl_model, env = train_models()

    # Initialize sound detection with correct model path and custom objects
    siren_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sounddetection', 'siren_classifier.h5')
    
    # Custom layer to handle the groups parameter
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)
    
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D
    }
    
    siren_classifier = load_model(siren_model_path, custom_objects=custom_objects, compile=False)
    
except Exception as e:
    print(f"Error initializing models: {e}")
    print("Please check if all dependencies are installed correctly.")
    sys.exit(1)

# Constants for coordinate conversion
BASE_LAT = 37.7749  # San Francisco latitude
BASE_LON = -122.4194  # San Francisco longitude
EARTH_RADIUS = 6371000  # Earth's radius in meters
SCALE_FACTOR = 5  # Scale factor for screen coordinates

# Sound detection configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_sound_detection():
    """Background thread for processing sound detection"""
    while True:
        try:
            filepath = sound_detection_queue.get()
            if filepath is None:
                break
                
            # Generate spectrogram
            spectrogram_path = generate_spectrogram(filepath)
            
            # Classify siren
            classification = classify_siren(filepath, siren_classifier)
            
            # Store results
            sound_detection_results.append({
                'timestamp': datetime.now().isoformat(),
                'filepath': filepath,
                'spectrogram': spectrogram_path,
                'classification': classification
            })
            
            # Keep only last 10 results
            if len(sound_detection_results) > 10:
                sound_detection_results.pop(0)
                
        except Exception as e:
            print(f"Error in sound detection: {e}")

# Start sound detection thread
sound_thread = threading.Thread(target=process_sound_detection, daemon=True)
sound_thread.start()

def screen_to_geo(x, y):
    """Convert screen coordinates to geographical coordinates."""
    # Calculate offsets in meters
    x_meters = x * SCALE_FACTOR
    y_meters = y * SCALE_FACTOR
    
    # Convert meters to degrees
    lat_offset = y_meters / (EARTH_RADIUS * np.pi / 180)
    lon_offset = x_meters / (EARTH_RADIUS * cos(radians(BASE_LAT)) * np.pi / 180)
    
    return BASE_LAT + lat_offset, BASE_LON + lon_offset

def geo_to_screen(lat, lon):
    """Convert geographical coordinates to screen coordinates."""
    # Calculate offsets in degrees
    lat_offset = lat - BASE_LAT
    lon_offset = lon - BASE_LON
    
    # Convert degrees to meters
    y_meters = lat_offset * EARTH_RADIUS * np.pi / 180
    x_meters = lon_offset * EARTH_RADIUS * cos(radians(BASE_LAT)) * np.pi / 180
    
    # Convert meters to screen coordinates
    x = x_meters / SCALE_FACTOR
    y = y_meters / SCALE_FACTOR
    
    return x, y

def update_blocked_roads():
    """Remove expired road blocks"""
    current_time = datetime.now()
    expired_blocks = []
    
    for road_id, block_info in blocked_roads.items():
        if block_info['until'] and block_info['until'] <= current_time:
            simulation.unblock_road(*block_info['segment'])
            expired_blocks.append(road_id)
    
    for road_id in expired_blocks:
        del blocked_roads[road_id]

def update_simulation_stats():
    """Update simulation statistics"""
    if simulation_stats['start_time'] is None:
        simulation_stats['start_time'] = datetime.now()
    
    # Update vehicle counts
    simulation_stats['total_vehicles'] = len(simulation.vehicles)
    simulation_stats['emergency_vehicles'] = len([v for v in simulation.vehicles if v.vehicle_type == VehicleType.EMERGENCY])
    
    # Calculate average wait time
    total_wait_time = sum(v.wait_time for v in simulation.vehicles)
    if simulation_stats['total_vehicles'] > 0:
        simulation_stats['avg_wait_time'] = total_wait_time / simulation_stats['total_vehicles']
    
    # Calculate system efficiency
    traffic_data = generate_traffic_data()
    total_efficiency = 0
    for signal_data in traffic_data.values():
        flow_ratio = signal_data['outgoing_flow'] / max(signal_data['incoming_flow'], 1)
        congestion_factor = 1 - signal_data['congestion_factor']
        total_efficiency += (flow_ratio + congestion_factor) / 2
    
    simulation_stats['system_efficiency'] = (total_efficiency / len(traffic_data)) * 100 if traffic_data else 0

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/api/upload-sound', methods=['POST'])
def upload_sound():
    """Handle sound file upload and detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Generate spectrogram
            spectrogram_path = generate_spectrogram(filepath)
            
            # Classify siren
            classification = classify_siren(filepath, siren_classifier)
            
            # Store result
            result = {
                'timestamp': datetime.now().isoformat(),
                'filepath': filepath,
                'spectrogram': spectrogram_path,
                'classification': classification
            }
            sound_detection_results.append(result)
            
            # Keep only last 10 results
            if len(sound_detection_results) > 10:
                sound_detection_results.pop(0)
            
            # If emergency siren detected, add emergency vehicle
            if classification == 'emergency_siren':
                # Add emergency vehicle to random signal
                target_signal = np.random.choice(list(SIGNALS.keys()))
                simulation.add_emergency_vehicle(target_signal)
            
            return jsonify({
                'success': True,
                'spectrogram': spectrogram_path,
                'classification': classification
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/sound-detection-results')
def get_sound_detection_results():
    """Get recent sound detection results."""
    return jsonify(sound_detection_results)

@app.route('/api/optimize-signals', methods=['POST'])
def optimize_signals():
    """Optimize traffic signals."""
    try:
        # Get optimization results
        results = predict_and_optimize(gnn_model, rl_model, env)
        
        # Apply optimization to simulation
        for signal_id, data in results.items():
            signal = next((s for s in simulation.grid_network['signals'] if s.id == signal_id), None)
            if signal:
                signal.congestion_level = data['congestion_factor']
                signal.cycle_time = data['traffic_data']['signal_timing']['green']
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimization-results')
def get_optimization_results():
    """Get latest optimization results."""
    # Get current signal states
    signal_states = {}
    for signal in simulation.grid_network['signals']:
        signal_states[signal.id] = {
            'congestion_level': signal.congestion_level,
            'cycle_time': signal.cycle_time,
            'throughput': signal.calculate_throughput(),
            'waiting_vehicles': len(signal.waiting_vehicles)
        }
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'results': signal_states
    })

@app.route('/api/stats')
def get_stats():
    """Get current simulation statistics."""
    update_simulation_stats()
    return jsonify(simulation_stats)

@app.route('/api/simulation-state')
def get_simulation_state():
    try:
        # Update simulation and blocked roads
        simulation.update()
        update_blocked_roads()
        update_simulation_stats()
        
        # Get vehicle positions and convert to geo coordinates
        vehicles = []
        for vehicle in simulation.vehicles:
            lat, lon = screen_to_geo(vehicle.x, vehicle.y)
            vehicles.append({
                'id': vehicle.id,
                'lat': float(lat),
                'lon': float(lon),
                'type': vehicle.vehicle_type.name,
                'color': vehicle.color,
                'is_emergency': vehicle.vehicle_type == VehicleType.EMERGENCY,
                'wait_time': vehicle.wait_time,
                'is_waiting': vehicle.is_waiting
            })
        
        # Get signal states
        signals = []
        for signal in simulation.grid_network['signals']:
            waiting_vehicles_count = len(signal.waiting_vehicles)
            throughput = signal.calculate_throughput()
            
            signals.append({
                'id': signal.id,
                'position': signal.real_coords,
                'is_green': signal.is_green,
                'congestion_level': float(signal.congestion_level),
                'waiting_vehicles': waiting_vehicles_count,
                'throughput': float(throughput),
                'cycle_time': signal.cycle_time,
                'current_time': signal.current_time
            })
        
        # Update simulation statistics
        simulation_stats.update({
            'total_waiting_vehicles': sum(len(s.waiting_vehicles) for s in simulation.grid_network['signals']),
            'average_wait_time': sum(v.wait_time for v in simulation.vehicles) / len(simulation.vehicles) if simulation.vehicles else 0,
            'total_throughput': sum(s.calculate_throughput() for s in simulation.grid_network['signals'])
        })
        
        return jsonify({
            'vehicles': vehicles,
            'signals': signals,
            'stats': simulation_stats,
            'blocked_roads': [{'start': list(start), 'end': list(end)} for start, end in simulation.blocked_roads]
        })
    except Exception as e:
        print(f"Error in simulation state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/traffic-data')
def get_traffic_data():
    try:
        traffic_data = generate_traffic_data()
        return jsonify(traffic_data)
    except Exception as e:
        print(f"Error getting traffic data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-emergency-vehicle', methods=['POST'])
def add_emergency_vehicle():
    """Add an emergency vehicle to the simulation."""
    try:
        data = request.get_json()
        vehicle_type = data.get('vehicle_type', 'ambulance')
        target_signal = data.get('target_signal')
        
        if not target_signal:
            return jsonify({'success': False, 'error': 'Target signal is required'})
        
        # Map vehicle type to VehicleType enum
        vehicle_type_map = {
            'ambulance': VehicleType.EMERGENCY,
            'fire': VehicleType.EMERGENCY,
            'police': VehicleType.EMERGENCY
        }
        
        vehicle_enum = vehicle_type_map.get(vehicle_type)
        if not vehicle_enum:
            return jsonify({'success': False, 'error': 'Invalid vehicle type'})
        
        # Add emergency vehicle to simulation
        success = simulation.add_emergency_vehicle(target_signal, vehicle_enum)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to add emergency vehicle'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/block-road', methods=['POST'])
def block_road():
    try:
        data = request.get_json()
        road_segment = data.get('road_segment')
        duration = data.get('duration', -1)  # -1 means indefinite
        
        if road_segment is not None:
            segments = simulation.grid_network['segments']
            if 0 <= road_segment < len(segments):
                segment = segments[road_segment]
                simulation.block_road(segment['start'], segment['end'])
                
                # Track the block duration
                block_until = None
                if duration > 0:
                    block_until = datetime.now() + timedelta(minutes=duration)
                
                blocked_roads[road_segment] = {
                    'until': block_until,
                    'segment': (segment['start'], segment['end'])
                }
                
                return jsonify({'status': 'success'})
        return jsonify({'error': 'Invalid road segment'}), 400
    except Exception as e:
        print(f"Error blocking road: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/unblock-road', methods=['POST'])
def unblock_road():
    try:
        data = request.get_json()
        road_segment = data.get('road_segment')
        
        if road_segment is not None:
            segments = simulation.grid_network['segments']
            if 0 <= road_segment < len(segments):
                segment = segments[road_segment]
                simulation.unblock_road(segment['start'], segment['end'])
                
                if road_segment in blocked_roads:
                    del blocked_roads[road_segment]
                
                return jsonify({'status': 'success'})
        return jsonify({'error': 'Invalid road segment'}), 400
    except Exception as e:
        print(f"Error unblocking road: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-simulation', methods=['POST'])
def reset_simulation():
    try:
        global simulation_stats, blocked_roads
        
        # Reset simulation
        simulation.reset()
        
        # Reset statistics
        simulation_stats = {
            'start_time': None,
            'total_vehicles': 0,
            'emergency_vehicles': 0,
            'avg_wait_time': 0,
            'system_efficiency': 0,
            'total_waiting_vehicles': 0,
            'average_wait_time': 0,
            'total_throughput': 0
        }
        
        # Clear blocked roads
        blocked_roads = {}
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error resetting simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        vehicle_count = data.get('vehicle_count')
        simulation_speed = data.get('simulation_speed')
        
        if vehicle_count is not None:
            simulation.set_vehicle_count(int(vehicle_count))
        
        if simulation_speed is not None:
            simulation.set_simulation_speed(float(simulation_speed))
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)