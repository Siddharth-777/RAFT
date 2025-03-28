from flask import Flask, render_template, jsonify, request, send_from_directory
import folium
import numpy as np
from traffic_data import SIGNALS, EDGES, generate_traffic_data, get_emergency_priority, EMERGENCY_PRIORITIES, get_traffic_flow
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import torch
from PIL import Image
import time
from model import train_models, predict_and_optimize

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'processed'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_map():
    m = folium.Map(location=[8.7833, 78.1333], zoom_start=13)  # Thoothukudi coordinates
    
    # Add traffic signals
    for signal, data in SIGNALS.items():
        # Determine color based on congestion
        if data['congestion_factor'] > 0.7:
            color = 'red'
        elif data['congestion_factor'] > 0.4:
            color = 'orange'
        else:
            color = 'green'
            
   
        popup_html = f"""
            <div style='font-family: Arial, sans-serif; min-width: 200px;'>
                <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>{data['name']}</h4>
                <div style='margin: 5px 0;'>
                    <strong>Congestion Level:</strong>
                    <div class="progress" style="height: 5px; margin-top: 5px;">
                        <div class="progress-bar bg-{color}" role="progressbar" 
                             style="width: {data['congestion_factor']*100}%"></div>
                    </div>
                    <small style='color: #6c757d;'>{data['congestion_factor']:.2f}</small>
                </div>
                <div style='margin: 5px 0;'>
                    <strong>Signal Timing:</strong><br>
                    <small>Green: {data['signal_timing']['green']}s | Red: {data['signal_timing']['red']}s</small>
                </div>
                <div style='margin: 5px 0;'>
                    <strong>Traffic Flow:</strong><br>
                    <small>Incoming: {data['incoming_flow']:.1f} | Outgoing: {data['outgoing_flow']:.1f}</small>
                </div>
                <div style='margin: 5px 0;'>
                    <strong>Emergency Frequency:</strong><br>
                    <small>{data['emergency_frequency']*100:.1f}%</small>
                </div>
                <div style='margin-top: 10px; font-size: 0.8em; color: #6c757d;'>
                    Last Updated: {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
        """
            
        folium.Marker(
            location=data['location'],
            popup=popup_html,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    

    for edge in EDGES:
        start = SIGNALS[edge[0]]['location']
        end = SIGNALS[edge[1]]['location']
        folium.PolyLine(
            locations=[start, end],
            weight=2,
            color='gray',
            opacity=0.5,
            dash_array='5'
        ).add_to(m)
    
    return m._repr_html_()

@app.route('/')
def index():
    try:
        map_html = create_map()
        return render_template('dashboard.html', map_html=map_html)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return str(e), 500

@app.route('/get_traffic_data')
def get_traffic_data():
    try:

        traffic_data = generate_traffic_data()
        
     
        stats = {
            'normal': 0,
            'moderate': 0,
            'heavy': 0
        }
        
        signals = []
        total_congestion = 0
        
        for signal, data in traffic_data.items():
            # Update statistics
            if data['congestion_factor'] > 0.7:
                stats['heavy'] += 1
            elif data['congestion_factor'] > 0.4:
                stats['moderate'] += 1
            else:
                stats['normal'] += 1
                
            total_congestion += data['congestion_factor']
            
            # Prepare signal data with additional metrics
            signals.append({
                'name': data['name'],
                'congestion_factor': data['congestion_factor'],
                'green_time': data['signal_timing']['green'],
                'red_time': data['signal_timing']['red'],
                'incoming_flow': data['incoming_flow'],
                'outgoing_flow': data['outgoing_flow'],
                'emergency_frequency': data['emergency_frequency'],
                'status': 'Heavy Traffic' if data['congestion_factor'] > 0.7 else 
                         'Moderate Congestion' if data['congestion_factor'] > 0.4 else 
                         'Normal Flow'
            })
        
        
        avg_congestion = total_congestion / len(signals) if signals else 0
        
        return jsonify({
            'stats': stats,
            'signals': signals,
            'avg_congestion': round(avg_congestion * 100, 2),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        print(f"Error in get_traffic_data route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_emergency', methods=['POST'])
def detect_emergency():
    try:
        
        traffic_data = generate_traffic_data()
        

        target_signal = list(traffic_data.keys())[0]
        signal_data = traffic_data[target_signal]
        
        
        frequency = signal_data['emergency_frequency']
        
     
        if frequency >= 0.7:
            priority = 'HIGH'
        elif frequency >= 0.4:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'
            
        
        priority_data = EMERGENCY_PRIORITIES[priority]
        

        response_time = np.random.randint(*priority_data['response_time'])
        clearance_time = np.random.randint(*priority_data['clearance_time'])
        
        return jsonify({
            'status': 'emergency',
            'priority': priority,
            'message': f'Emergency vehicle detected at {signal_data["name"]}!',
            'response_time': f'{response_time}s',
            'clearance_time': f'{clearance_time}s',
            'clearance_status': f'Emergency Clearance ({priority} Priority)',
            'signal_name': signal_data["name"],
            'frequency': f'{frequency*100:.1f}%',
            'congestion': f'{signal_data["congestion_factor"]*100:.1f}%',
            'incoming_flow': f'{signal_data["incoming_flow"]:.1f}',
            'outgoing_flow': f'{signal_data["outgoing_flow"]:.1f}'
        })
    except Exception as e:
        print(f"Error in detect_emergency route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get traffic data for the signal
            traffic_data = generate_traffic_data()
            target_signal = list(traffic_data.keys())[0]
            signal_data = traffic_data[target_signal]
            
            
            congestion_factor = signal_data['congestion_factor']
            incoming_flow = signal_data['incoming_flow']
            outgoing_flow = signal_data['outgoing_flow']
            emergency_frequency = signal_data['emergency_frequency']
            
          
            if congestion_factor > 0.7:
                traffic_status = 'Heavy Traffic'
            elif congestion_factor > 0.4:
                traffic_status = 'Moderate Congestion'
            else:
                traffic_status = 'Normal Flow'
            
      
            emergency_threshold = 0.6 if traffic_status == 'Heavy Traffic' else 0.8
            is_emergency = (os.path.getsize(filepath) > 100000 and 
                          np.random.random() < emergency_threshold)
            
            if is_emergency:
             
                if congestion_factor > 0.7 and emergency_frequency > 0.6:
                    priority = 'HIGH'
                elif congestion_factor > 0.4 or emergency_frequency > 0.4:
                    priority = 'MEDIUM'
                else:
                    priority = 'LOW'
                    
               
                priority_data = EMERGENCY_PRIORITIES[priority]
                
              
                base_response_time = np.random.randint(*priority_data['response_time'])
                base_clearance_time = np.random.randint(*priority_data['clearance_time'])
                
             
                traffic_multiplier = 1.5 if traffic_status == 'Heavy Traffic' else 1.2
                response_time = int(base_response_time * traffic_multiplier)
                clearance_time = int(base_clearance_time * traffic_multiplier)
                
                return jsonify({
                    'status': 'emergency',
                    'priority': priority,
                    'message': f'Emergency vehicle detected at {signal_data["name"]}! ({traffic_status})',
                    'response_time': f'{response_time}s',
                    'clearance_time': f'{clearance_time}s',
                    'clearance_status': f'Emergency Clearance ({priority} Priority)',
                    'signal_name': signal_data["name"],
                    'frequency': f'{emergency_frequency*100:.1f}%',
                    'congestion': f'{congestion_factor*100:.1f}%',
                    'incoming_flow': f'{incoming_flow:.1f}',
                    'outgoing_flow': f'{outgoing_flow:.1f}',
                    'traffic_status': traffic_status
                })
            else:
                return jsonify({
                    'status': 'normal',
                    'message': f'No emergency vehicle detected. Current traffic: {traffic_status}',
                    'signal_name': signal_data["name"],
                    'frequency': f'{emergency_frequency*100:.1f}%',
                    'congestion': f'{congestion_factor*100:.1f}%',
                    'incoming_flow': f'{incoming_flow:.1f}',
                    'outgoing_flow': f'{outgoing_flow:.1f}',
                    'traffic_status': traffic_status
                })
                
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"Error in upload_audio route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_vehicles', methods=['POST'])
def detect_vehicles():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})
    
    try:
       
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
 
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return jsonify({'error': 'Error reading video file'})
        else:
            frame = cv2.imread(filepath)
            if frame is None:
                return jsonify({'error': 'Error reading image file'})
        

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        start_time = time.time()
        results = yolo_model(frame_rgb)
        processing_time = int((time.time() - start_time) * 1000)  
        
   
        cars_detected = 0
        bikes_detected = 0
        heavy_vehicles = 0
        
        for detection in results.xyxy[0]:
            class_id = int(detection[5])
            if class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                if class_id in [2, 3]:  # car
                    cars_detected += 1
                elif class_id == 3:  # motorcycle
                    bikes_detected += 1
                else:  # bus, truck
                    heavy_vehicles += 1
        

        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', processed_filename)
        
  
        results_image = Image.fromarray(results.render()[0])
        results_image.save(processed_path)
        
        return jsonify({
            'cars_detected': cars_detected,
            'bikes_detected': bikes_detected,
            'heavy_vehicles': heavy_vehicles,
            'processing_time': processing_time,
            'output_url': f'/uploads/processed/{processed_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/uploads/processed/<filename>')
def processed_file(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), filename)

# Load YOLO model
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

yolo_model = load_yolo_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
