from yolo import detect_ambulance, detect_all_vehicles


video_path = "videos/ambulance.mp4"
detect_ambulance(video_path)

video_path = "videos/indian_traffic.mp4"
detect_all_vehicles(video_path)