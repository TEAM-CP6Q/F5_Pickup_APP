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

# 학습된 커스텀 모델 경로 설정 (이 부분을 실제 모델 경로로 변경)
CUSTOM_MODEL_PATH = "runs/continue_train/continued_model3/weights/best.pt"
# 백업용 기본 모델 (커스텀 모델 로드 실패시 사용)
DEFAULT_MODEL_NAME = "yolov8n.pt"

# 대구시 대형폐기물 가격 데이터 (API에 없는 품목에 대한 백업)
DAEGU_WASTE_PRICES = {
    "laptop": 2000,           # 컴퓨터, 노트북 모든규격 - 2,000원
    "tv_small": 3000,         # T.V 42인치미만 - 3,000원
    "tv_large": 6000,         # T.V 42인치이상 - 6,000원
    "refrigerator_small": 5000,  # 냉장고 300ℓ미만 - 5,000원
    "refrigerator_medium": 7000, # 냉장고 300ℓ이상 500ℓ미만 - 7,000원
    "refrigerator_large": 9000,  # 냉장고 500ℓ이상 700ℓ미만 - 9,000원
    "refrigerator_xlarge": 15000, # 냉장고 700ℓ이상 - 15,000원
    "microwave": 3000,        # 전자렌지 모든규격 - 3,000원
    "oven": 5000,             # 전기오븐렌지 높이 1m 이상 - 5,000원
    "cell_phone": 1000,       # 소형가전으로 분류 - 1,000원
    "keyboard": 1000,         # 소형가전으로 분류 - 1,000원
    "mouse": 1000,            # 소형가전으로 분류 - 1,000원
    "remote": 1000,           # 소형가전으로 분류 - 1,000원
    "toaster": 1000,          # 소형가전으로 분류 - 1,000원
    "hair_drier": 1000,       # 소형가전으로 분류 - 1,000원
    "clock": 1000,            # 소형가전으로 분류 - 1,000원
    "chair_small": 2000,      # 의자 소형 - 2,000원
    "chair_large": 4000,      # 의자 대형 - 4,000원
    "couch_single": 4000,     # 쇼파 1인용 - 4,000원
    "couch_double": 6000,     # 쇼파 2인용 (1인 추가시 2,000원) - 6,000원
    "bed_single": 6000,       # 침대 1인용 구조물 - 6,000원
    "bed_double": 8000,       # 침대 2인용 구조물 - 8,000원
    "dining_table_small": 4000, # 식탁 6인용 미만 - 4,000원
    "dining_table_large": 6000, # 식탁 6인용 이상 - 6,000원
    "toilet": 5000,           # 변기 좌변기 - 5,000원
    "sink": 5000,             # 세면대 - 5,000원
    "vase": 1000,             # 화분(소형) - 1,000원
    "scissors": 1000,         # 소형가전으로 분류 - 1,000원
    "book": 1000,             # 책 - 1,000원
    "potted_plant_small": 1000, # 화분 소형 - 1,000원
    "potted_plant_large": 2000, # 화분 대형 - 2,000원
    "toothbrush": 1000,       # 일반쓰레기 소형 - 1,000원
    "teddy_bear": 1000,       # 장난감 kg당 - 1,000원
    # 소형 전자제품
    "airpods": 1000,          # 소형가전으로 분류 - 1,000원
    "smartwatch": 1000,       # 소형가전으로 분류 - 1,000원
    "earphone": 1000,         # 소형가전으로 분류 - 1,000원
    "headphone": 1000,        # 소형가전으로 분류 - 1,000원
    "tablet": 2000,           # 컴퓨터, 노트북 분류 - 2,000원
    "speaker": 2000,          # 소형가전으로 분류 - 2,000원
    "camera": 1000,           # 소형가전으로 분류 - 1,000원
}

# 커스텀 모델에서 감지한 레이블을 API 카테고리에 매핑
CUSTOM_TO_WASTE_MAP = {
    # 재활용품 카테고리
    "can": {"category": "재활용품", "type": "캔", "price": 1000},
    "paper": {"category": "재활용품", "type": "종이", "price": 1000},
    "plastic": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    "glass": {"category": "재활용품", "type": "유리", "price": 1000},
    "metal": {"category": "재활용품", "type": "금속", "price": 1000},
    "vinyl": {"category": "재활용품", "type": "비닐", "price": 1000},
    "cardboard": {"category": "재활용품", "type": "종이", "price": 1000},
    "newspaper": {"category": "재활용품", "type": "종이", "price": 1000},
    "bottle": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    "pet": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    
    # 일반쓰레기 카테고리
    "food": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "rubber": {"category": "일반쓰레기", "type": "기타", "price": 1000},
    "tissue": {"category": "일반쓰레기", "type": "기타", "price": 1000},
    "cigarette": {"category": "일반쓰레기", "type": "기타", "price": 1000},
    
    # 가전제품/전자제품 카테고리
    "laptop": {"category": "가전제품", "type": "컴퓨터, 노트북", "price": 2000},
    "tv": {"category": "가전제품", "type": "T.V", "price": 3000},
    "smartphone": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "computer": {"category": "가전제품", "type": "컴퓨터, 노트북", "price": 2000},
    "refrigerator": {"category": "가전제품", "type": "냉장고", "price": 7000},
    "microwave": {"category": "가전제품", "type": "전자렌지", "price": 3000},
    "washing_machine": {"category": "가전제품", "type": "세탁기", "price": 5000},
    "air_conditioner": {"category": "가전제품", "type": "에어컨", "price": 3000},
    "vacuum": {"category": "가전제품", "type": "청소기", "price": 1000},
    
    # 가구류 카테고리
    "chair": {"category": "가구류", "type": "의자", "price": 2000},
    "desk": {"category": "가구류", "type": "책상", "price": 6000},
    "table": {"category": "가구류", "type": "식탁", "price": 4000},
    "sofa": {"category": "가구류", "type": "쇼파", "price": 4000},
    "bed": {"category": "가구류", "type": "침대", "price": 6000},
    "cabinet": {"category": "가구류", "type": "서랍장", "price": 3000},
    "bookshelf": {"category": "가구류", "type": "책장", "price": 4000},
    
    # 기본 매핑
    "pack": {"category": "재활용품", "type": "기타", "price": 1000}
}

# COCO 데이터셋 클래스 중 폐기물 관련 클래스 매핑 (백업용)
COCO_TO_WASTE_MAP = {
    # 일반 재활용품 카테고리
    "bottle": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    "wine glass": {"category": "재활용품", "type": "유리", "price": 1000},
    "cup": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    "fork": {"category": "재활용품", "type": "금속", "price": 1000},
    "knife": {"category": "재활용품", "type": "금속", "price": 1000},
    "spoon": {"category": "재활용품", "type": "금속", "price": 1000},
    "bowl": {"category": "재활용품", "type": "플라스틱", "price": 1000},
    "banana": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "apple": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "sandwich": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "orange": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "broccoli": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "carrot": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "hot dog": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "pizza": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "donut": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "cake": {"category": "일반쓰레기", "type": "음식물", "price": 1000},
    "chair": {"category": "가구류", "type": "의자", "price": 2000},
    "couch": {"category": "가구류", "type": "쇼파", "price": 4000},
    "potted plant": {"category": "기타류", "type": "화분", "price": 1000},
    "bed": {"category": "가구류", "type": "침대", "price": 6000},
    "dining table": {"category": "가구류", "type": "식탁", "price": 4000},
    "toilet": {"category": "욕실용품", "type": "변기", "price": 5000},
    "tv": {"category": "가전제품", "type": "T.V", "price": 3000},
    "laptop": {"category": "가전제품", "type": "컴퓨터, 노트북", "price": 2000},
    "mouse": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "remote": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "keyboard": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "cell phone": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "microwave": {"category": "가전제품", "type": "전자렌지", "price": 3000},
    "oven": {"category": "가전제품", "type": "전기오븐렌지", "price": 5000},
    "toaster": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "sink": {"category": "욕실용품", "type": "세면대", "price": 5000},
    "refrigerator": {"category": "가전제품", "type": "냉장고", "price": 7000},
    "book": {"category": "재활용품", "type": "종이", "price": 1000},
    "clock": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "vase": {"category": "기타류", "type": "화분", "price": 1000},
    "scissors": {"category": "재활용품", "type": "금속", "price": 1000},
    "teddy bear": {"category": "유아용품", "type": "장난감", "price": 1000},
    "hair drier": {"category": "가전제품", "type": "기타 소형가전", "price": 1000},
    "toothbrush": {"category": "일반쓰레기", "type": "기타", "price": 1000}
}

# YOLO 모델 로드 - Ultralytics 라이브러리 사용
def load_model():
    try:
        # 학습된 커스텀 모델 로드 시도
        if os.path.exists(CUSTOM_MODEL_PATH):
            print(f"커스텀 모델 로드 중: {CUSTOM_MODEL_PATH}")
            model = YOLO(CUSTOM_MODEL_PATH)
            print(f"커스텀 모델 로드 성공")
            # 모델의 클래스 목록 확인
            print(f"감지 가능한 클래스: {model.names}")
            return model
        else:
            # 커스텀 모델이 없는 경우 기본 모델 로드
            print(f"커스텀 모델을 찾을 수 없습니다. 기본 모델 로드 중: {DEFAULT_MODEL_NAME}")
            model = YOLO(DEFAULT_MODEL_NAME)
            print(f"기본 모델 로드 성공: {DEFAULT_MODEL_NAME}")
            return model
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        try:
            # 오류 발생 시 기본 모델 시도
            print(f"기본 모델 로드 시도 중: {DEFAULT_MODEL_NAME}")
            model = YOLO(DEFAULT_MODEL_NAME)
            print(f"기본 모델 로드 성공: {DEFAULT_MODEL_NAME}")
            return model
        except Exception as e2:
            print(f"기본 모델 로딩 오류: {e2}")
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
        results = model(temp_image_path, conf=0.25)  # 낮은 신뢰도 임계값 사용
        
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
                
                # 클래스 매핑 및 정보 추출
                waste_info = get_waste_info(class_name, image, x1, y1, x2, y2)
                
                detections.append({
                    "class_name": class_name,
                    "category": waste_info["category"],
                    "type": waste_info["type"],
                    "price": waste_info.get("price", 1000),  # 기본가 1000원
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections
    except Exception as e:
        print(f"객체 감지 오류: {e}")
        return []

# 객체의 클래스에 따라 적절한 매핑 제공
def get_waste_info(class_name, image, x1, y1, x2, y2):
    # 우선순위: 커스텀 매핑 -> COCO 매핑 -> 기본값
    
    # 1. 커스텀 매핑에서 검색
    if class_name in CUSTOM_TO_WASTE_MAP:
        waste_info = CUSTOM_TO_WASTE_MAP[class_name].copy()
        
        # 병 종류 특수 처리
        if class_name == "bottle" or class_name.lower() == "pet":
            waste_type = analyze_bottle_type(image, x1, y1, x2, y2)
            waste_info["type"] = waste_type
        
        return waste_info
    
    # 2. COCO 매핑에서 검색
    if class_name in COCO_TO_WASTE_MAP:
        waste_info = COCO_TO_WASTE_MAP[class_name].copy()
        
        # 병 종류 특수 처리
        if class_name == "bottle":
            waste_type = analyze_bottle_type(image, x1, y1, x2, y2)
            waste_info["type"] = waste_type
        
        return waste_info
    
    # 3. 기본값 반환
    print(f"매핑되지 않은 클래스: {class_name}, 기본값 사용")
    return {
        "category": "일반쓰레기",
        "type": "기타",
        "price": 1000
    }

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
            default_price = detection.get("price", 1000)
            
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
                    print(f"API에서 '{category}' 카테고리의 '{waste_type}' 타입을 찾을 수 없습니다. 기본 가격 사용")
                    # API에 없는 경우 기본가격 사용
                    mapped_items.append({
                        "category": category,
                        "type": waste_type,
                        "description": f"{waste_type} 폐기물",
                        "price": default_price,
                        "quantity": 1,
                        "totalPrice": default_price,
                        "confidence": round(detection["confidence"] * 100)
                    })
            else:
                print(f"API에서 '{category}' 카테고리를 찾을 수 없습니다. 기본 가격 사용")
                # API에 없는 경우 기본가격 사용
                mapped_items.append({
                    "category": category,
                    "type": waste_type,
                    "description": f"{waste_type} 폐기물",
                    "price": default_price,
                    "quantity": 1,
                    "totalPrice": default_price,
                    "confidence": round(detection["confidence"] * 100)
                })
        
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
            model_type = "custom" if os.path.exists(CUSTOM_MODEL_PATH) else "default"
            model_path = CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else DEFAULT_MODEL_NAME
            
            return jsonify({
                "model_name": model_path,
                "model_type": model_type,
                "classes": app.model.names,
                "status": "loaded"
            })
        else:
            return jsonify({
                "model_name": CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else DEFAULT_MODEL_NAME,
                "model_type": "not_loaded",
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