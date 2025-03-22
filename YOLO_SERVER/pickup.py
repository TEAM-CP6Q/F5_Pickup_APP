import os
import io
import base64
import numpy as np
import cv2
import json
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ultralytics 라이브러리 사용 (PyTorch 대체)
from ultralytics import YOLO

# waste_detection_server.py 파일 수정
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB로 설정
CORS(app)

# 환경 변수 설정
WASTE_API_URL = "https://refresh-f5-server.o-r.kr/api/pickup/waste/type-list"
MODEL_NAME = "yolov8n.pt"  # yolov8 nano 모델 (더 작고 빠른 모델)

# COCO 데이터셋 클래스 중 폐기물 관련 클래스 매핑
COCO_TO_WASTE_MAP = {
    # 일반 재활용품 카테고리
    "bottle": {"category": "재활용품", "type": "플라스틱"},  # 기본적으로 병은 플라스틱으로 가정
    "wine glass": {"category": "재활용품", "type": "유리"},
    "cup": {"category": "재활용품", "type": "플라스틱"},
    "fork": {"category": "재활용품", "type": "금속"},
    "knife": {"category": "재활용품", "type": "금속"},
    "spoon": {"category": "재활용품", "type": "금속"},
    "bowl": {"category": "재활용품", "type": "플라스틱"},
    "banana": {"category": "일반쓰레기", "type": "음식물"},
    "apple": {"category": "일반쓰레기", "type": "음식물"},
    "sandwich": {"category": "일반쓰레기", "type": "음식물"},
    "orange": {"category": "일반쓰레기", "type": "음식물"},
    "broccoli": {"category": "일반쓰레기", "type": "음식물"},
    "carrot": {"category": "일반쓰레기", "type": "음식물"},
    "hot dog": {"category": "일반쓰레기", "type": "음식물"},
    "pizza": {"category": "일반쓰레기", "type": "음식물"},
    "donut": {"category": "일반쓰레기", "type": "음식물"},
    "cake": {"category": "일반쓰레기", "type": "음식물"},
    "chair": {"category": "생활용품", "type": "가구"},
    "couch": {"category": "생활용품", "type": "가구"},
    "potted plant": {"category": "일반쓰레기", "type": "기타"},
    "bed": {"category": "생활용품", "type": "가구"},
    "dining table": {"category": "생활용품", "type": "가구"},
    "toilet": {"category": "생활용품", "type": "변기"},
    "tv": {"category": "생활용품", "type": "도퍼 매트"},
    "laptop": {"category": "생활용품", "type": "전자제품"},
    "mouse": {"category": "생활용품", "type": "전자제품"},
    "remote": {"category": "생활용품", "type": "전자제품"},
    "keyboard": {"category": "생활용품", "type": "전자제품"},
    "cell phone": {"category": "생활용품", "type": "전자제품"},
    "microwave": {"category": "생활용품", "type": "도퍼 매트"},
    "oven": {"category": "생활용품", "type": "도퍼 매트"},
    "toaster": {"category": "생활용품", "type": "전자제품"},
    "sink": {"category": "생활용품", "type": "기타"},
    "refrigerator": {"category": "생활용품", "type": "도퍼 매트"},
    "book": {"category": "재활용품", "type": "종이"},
    "clock": {"category": "생활용품", "type": "전자제품"},
    "vase": {"category": "재활용품", "type": "유리"},
    "scissors": {"category": "재활용품", "type": "금속"},
    "teddy bear": {"category": "생활용품", "type": "기타"},
    "hair drier": {"category": "생활용품", "type": "전자제품"},
    "toothbrush": {"category": "일반쓰레기", "type": "기타"}
}

# YOLO 모델 로드 - Ultralytics 라이브러리 사용
def load_model():
    try:
        # Ultralytics YOLO 모델 로드
        model = YOLO(MODEL_NAME)
        print(f"모델 로드 성공: {MODEL_NAME}")
        return model
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        return None

# 폐기물 API에서 데이터 가져오기
def fetch_waste_types():
    try:
        response = requests.get(WASTE_API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API 오류: {response.status_code}")
            return {}
    except Exception as e:
        print(f"API 요청 오류: {e}")
        return {}

# 이미지 처리 및 객체 인식 (Ultralytics YOLO 사용)
def detect_waste(image_data, model):
    try:
        # 이미지 처리
        image = Image.open(io.BytesIO(image_data))
        
        # 이미지 저장 (임시)
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)
        
        # YOLO 모델로 객체 감지
        results = model(temp_image_path)
        
        # 임시 파일 삭제
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # 결과 추출
        detections = []
        
        # 결과 반복 처리
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 바운딩 박스 좌표
                conf = float(box.conf[0])  # 신뢰도
                cls = int(box.cls[0])  # 클래스 인덱스
                class_name = model.names[cls]  # 클래스 이름
                
                print(f"감지된 객체: {class_name}, 신뢰도: {conf:.2f}")
                
                # COCO 클래스명과 매핑
                if class_name in COCO_TO_WASTE_MAP:
                    waste_info = COCO_TO_WASTE_MAP[class_name]
                    
                    # 병 종류 추가 분석 (유리/플라스틱 구분)
                    if class_name == "bottle":
                        waste_type = analyze_bottle_type(image, x1, y1, x2, y2)
                        waste_info = {"category": "재활용품", "type": waste_type}
                    
                    detections.append({
                        "class_name": class_name,
                        "category": waste_info["category"],
                        "type": waste_info["type"],
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        return detections
    except Exception as e:
        print(f"객체 감지 오류: {e}")
        return []

# 병 종류 분석 (유리병/플라스틱병 구분)
def analyze_bottle_type(image, xmin, ymin, xmax, ymax):
    try:
        # 바운딩 박스 좌표를 정수로 변환
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # 이미지를 numpy 배열로 변환
        img_array = np.array(image)
        
        # 바운딩 박스 영역 추출
        bottle_region = img_array[ymin:ymax, xmin:xmax]
        
        if bottle_region.size == 0:
            return "플라스틱"  # 기본값
        
        # HSV 색공간으로 변환
        if len(bottle_region.shape) == 3:  # RGB 이미지 확인
            bottle_hsv = cv2.cvtColor(bottle_region, cv2.COLOR_RGB2HSV)
            
            # 유리병 특성: 투명도가 높고 채도가 낮은 경향이 있음
            # 플라스틱병 특성: 채도가 높거나 선명한 색상을 가진 경향이 있음
            
            # 채도(Saturation) 분석
            avg_saturation = np.mean(bottle_hsv[:, :, 1])
            
            # 밝기(Value) 분석
            avg_value = np.mean(bottle_hsv[:, :, 2])
            
            # 색상 편차(표준편차) 분석 - 유리병은 주변 색상이 더 균일한 경향
            color_std = np.std(bottle_region)
            
            print(f"병 분석 - 채도: {avg_saturation:.2f}, 밝기: {avg_value:.2f}, 색상 편차: {color_std:.2f}")
            
            # 간단한 분류 규칙 (실제 애플리케이션에서는 더 정교한 알고리즘 사용 필요)
            if avg_saturation < 50 and color_std < 40:
                return "유리"
            else:
                return "플라스틱"
        else:
            return "플라스틱"  # 기본값
    except Exception as e:
        print(f"병 종류 분석 오류: {e}")
        return "플라스틱"  # 기본값

# 인식된 폐기물을 API 데이터와 매핑
def map_to_waste_types(detections, waste_types):
    try:
        mapped_items = []
        for detection in detections:
            category = detection["category"]
            waste_type = detection["type"]
            
            # 카테고리가 API에 존재하는지 확인
            if category in waste_types and waste_types[category]:
                # 해당 카테고리에서 일치하는 폐기물 유형 찾기
                matching_items = [item for item in waste_types[category] 
                                 if item["type"] == waste_type]
                
                if matching_items:
                    item = matching_items[0]  # 첫 번째 일치 항목 사용
                    mapped_items.append({
                        "category": category,
                        "type": waste_type,
                        "description": item.get("description", ""),
                        "price": item["price"],
                        "quantity": 1,
                        "totalPrice": item["price"],
                        "confidence": round(detection["confidence"] * 100)
                    })
                else:
                    print(f"API에서 '{category}' 카테고리의 '{waste_type}' 타입을 찾을 수 없습니다.")
            else:
                print(f"API에서 '{category}' 카테고리를 찾을 수 없습니다.")
        
        # 중복 제거 (같은 유형은 수량 합산)
        result = {}
        for item in mapped_items:
            key = f"{item['category']}-{item['type']}"
            if key not in result:
                result[key] = item
            else:
                result[key]["quantity"] += 1
                result[key]["totalPrice"] = result[key]["price"] * result[key]["quantity"]
        
        return list(result.values())
    except Exception as e:
        print(f"폐기물 매핑 오류: {e}")
        return []

# API 엔드포인트: 폐기물 인식
@app.route('/api/detect-waste', methods=['POST'])
def detect_waste_api():
    try:
        # 이미지 데이터 받기
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({"error": "이미지가 제공되지 않았습니다"}), 400
        
        # 파일 또는 base64 형식 처리
        if 'image' in request.files:
            image_data = request.files['image'].read()
        else:
            # base64 디코딩
            image_base64 = request.form['image']
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            image_data = base64.b64decode(image_base64)
        
        # 폐기물 유형 데이터 가져오기
        waste_types = fetch_waste_types()
        
        # 모델이 로드되지 않은 경우 로드
        if not hasattr(app, 'model'):
            app.model = load_model()
            
        if app.model is None:
            return jsonify({"error": "모델을 로드할 수 없습니다"}), 500
        
        # 폐기물 인식 (실제 YOLO 모델 사용)
        detections = detect_waste(image_data, app.model)
        
        # 인식된 객체가 없는 경우
        if not detections:
            return jsonify({
                "success": False,
                "message": "이미지에서 폐기물을 인식하지 못했습니다.",
                "detections": [],
                "mapped_items": []
            })
        
        # API 데이터와 매핑
        mapped_items = map_to_waste_types(detections, waste_types)
        
        # 결과 반환
        return jsonify({
            "success": True,
            "detections": detections,
            "mapped_items": mapped_items
        })
    
    except Exception as e:
        print(f"API 오류: {e}")
        return jsonify({"error": str(e)}), 500

# 모델 정보 반환 API
@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        if hasattr(app, 'model'):
            return jsonify({
                "model_name": MODEL_NAME,
                "status": "loaded"
            })
        else:
            return jsonify({
                "model_name": MODEL_NAME,
                "status": "not_loaded"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 애플리케이션 초기화
if __name__ == '__main__':
    # 모델 미리 로드
    app.model = load_model()
    
    # 서버 실행
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)