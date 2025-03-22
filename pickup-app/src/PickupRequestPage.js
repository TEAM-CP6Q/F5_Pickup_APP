import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  Modal,
  Animated,
  Dimensions,
  Platform,
  ActivityIndicator,
  Image,
  StatusBar,
  SafeAreaView
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useNavigation } from "@react-navigation/native";
import Postcode from '@actbase/react-daum-postcode';
import { CameraView } from 'expo-camera';
import * as Camera from 'expo-camera'; 
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import styles from "../styles/PickupRequestStyles";

// 상단바 높이 계산
const STATUSBAR_HEIGHT = Platform.OS === 'ios' ? 44 : StatusBar.currentHeight || 24;

// 스타일 확장
const extendedStyles = {
  ...styles,
  safeArea: {
    flex: 1,
    backgroundColor: "#fff",
  },
  mainContainer: {
    ...styles.mainContainer,
    paddingTop: Platform.OS === 'android' ? STATUSBAR_HEIGHT : 0,
  },
  scrollContent: {
    ...styles.scrollContent,
    paddingTop: 10, // 상단 여백 추가
  }
};

const { width } = Dimensions.get('window');

// 서버 API URL
const WASTE_API_URL = "https://refresh-f5-server.o-r.kr/api/pickup/waste/type-list";
// YOLO 서버 URL을 실제 서버 URL로 설정 (배포 시 변경 필요)
const WASTE_DETECTION_API_URL = "http://172.30.1.87:8080/api/detect-waste";

const PickupRequestPage = () => {
  const navigation = useNavigation();
  const [currentScreen, setCurrentScreen] = useState(1);
  const [slideDirection, setSlideDirection] = useState("none");
  const [isOpen, setIsOpen] = useState(false); // 주소검색 모달 상태 추가

  // 애니메이션 값 추가
  const slideAnim = useState(new Animated.Value(0))[0];

  // Calendar state
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [selectedTime, setSelectedTime] = useState(null);
  const [currentMonth, setCurrentMonth] = useState(new Date().getMonth());
  const [currentYear, setCurrentYear] = useState(new Date().getFullYear());

  // Second screen state with default user data
  const [userData, setUserData] = useState("");

  // 카메라 및 분석 관련 상태
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraVisible, setCameraVisible] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recognizedItems, setRecognizedItems] = useState([]);
  const [capturedImage, setCapturedImage] = useState(null);
  const cameraRef = useRef(null);

  // 폐기물 관련 state
  const [wasteTypes, setWasteTypes] = useState({}); // 전체 폐기물 타입 데이터
  const [selectedItems, setSelectedItems] = useState([]); // 선택된 폐기물 목록
  const [currentWaste, setCurrentWaste] = useState({ // 현재 선택된 폐기물 정보
    category: "",
    item: null,
    quantity: 1
  });
  const [totalPrice, setTotalPrice] = useState(0);
  
  // 디버그 모드 - 실제 배포 시 false로 설정
  const [debugMode, setDebugMode] = useState(false);

  // 주소 검색 완료 핸들러 추가
  const completeHandler = (data) => {
    setUserData(prev => ({
      ...prev,
      postalCode: data.zonecode,
      roadNameAddress: data.roadAddress
    }));
    setIsOpen(false);
  };

  // 사용자 정보 가져오기
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const email = await AsyncStorage.getItem('email');
        const token = await AsyncStorage.getItem('token');

        if (!email || !token) {
          console.error('이메일 또는 토큰이 없습니다.');
          return;
        }

        const response = await fetch(`https://refresh-f5-server.o-r.kr/api/account/search-account/${email}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });

        if (response.status === 200) {
          const data = await response.json();
          setUserData({
            name: data.name,
            phoneNumber: data.phoneNumber,
            email: email,
            postalCode: data.postalCode || '',
            roadNameAddress: data.roadNameAddress || '',
            detailedAddress: data.detailedAddress || ''
          });
        } else {
          console.error('사용자 정보를 가져오는데 실패했습니다.');
        }
      } catch (error) {
        console.error('API 호출 중 에러 발생:', error);
      }
    };

    fetchUserData();
  }, []);

  // 폐기물 타입 데이터 불러오기
  useEffect(() => {
    const fetchWasteTypes = async () => {
      try {
        const response = await fetch(WASTE_API_URL, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          const data = await response.json();
          setWasteTypes(data);
          console.log("폐기물 타입 데이터 로드 성공:", data);
        }
      } catch (error) {
        console.error('폐기물 타입 데이터 로드 실패:', error);
      }
    };

    fetchWasteTypes();
  }, []);

  // 카메라 권한 요청 및 처리
  const handleOpenCamera = async () => {
    try {
      // 권한 요청 로직 간소화
      setHasPermission(true); // 권한 우회 설정
      setCameraVisible(true);
    } catch (error) {
      console.error("카메라 열기 실패:", error);
      
      // 오류 발생 시에도 카메라 모달 표시 시도
      setHasPermission(true);
      setCameraVisible(true);
    }
  };

  // 이미지 캡처 및 분석
  const handleCapture = async () => {
    if (!cameraRef.current) return;
    
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      setCapturedImage(photo.uri);
      setCameraVisible(false);
      
      // 이미지 분석 시작
      analyzeImage(photo.uri);
    } catch (e) {
      console.error("사진 촬영 오류:", e);
      Alert.alert('오류', '사진 촬영 중 오류가 발생했습니다.');
    }
  };

  // 이미지 분석 함수 - YOLO 서버 API 사용
  const analyzeImage = async (uri) => {
    setIsAnalyzing(true);

    try {
      console.log("폐기물 인식 시작...");
      
      // 이미지 리사이징 및 압축 (서버 전송 최적화)
      const resizedImage = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 600, height: 600 } }], // 크기를 더 작게 조정
        { format: 'jpeg', base64: true, compress: 0.5 } // 압축률 50%로 설정
      );
      
      console.log("이미지 리사이징 완료");

      if (debugMode) {
        // 디버그 모드: 서버 호출 없이 더미 데이터 사용
        console.log("디버그 모드: 더미 데이터 사용");
        setTimeout(() => {
          const dummyResponse = {
            success: true,
            mapped_items: [
              {
                category: "재활용품",
                type: "플라스틱",
                description: "페트병",
                price: 500,
                quantity: 1,
                totalPrice: 500,
                confidence: 85
              }
            ]
          };
          setRecognizedItems(dummyResponse.mapped_items);
          setIsAnalyzing(false);
        }, 2000);
        return;
      }

      // FormData 준비
      const formData = new FormData();
      formData.append('image', resizedImage.base64);

      // 서버에 이미지 전송 및 분석 요청
      const response = await fetch(WASTE_DETECTION_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`서버 응답 오류: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success && result.mapped_items && result.mapped_items.length > 0) {
        console.log("서버 인식 결과:", result);
        
        // 인식된 아이템 설정
        setRecognizedItems(result.mapped_items);
        
        // 추가 디버깅 정보
        if (result.detections) {
          console.log("감지된 객체:", result.detections.map(d => 
            `${d.class_name} (${d.type}): ${Math.round(d.confidence * 100)}%`
          ).join(', '));
        }
      } else {
        console.log("인식 결과 없음 또는 오류:", result);
        
        // 사용자에게 알림
        Alert.alert(
          "인식 실패", 
          "폐기물을 인식하지 못했습니다. 다시 시도하거나 직접 선택해주세요.",
          [{ text: "확인" }]
        );
        
        // 인식된 아이템 초기화
        setRecognizedItems([]);
      }
      
    } catch (error) {
      console.error("이미지 분석 실패:", error);
      
      if (debugMode) {
        // 디버그 모드: 오류 발생해도 더미 데이터 사용
        const dummyResponse = {
          success: true,
          mapped_items: [
            {
              category: "재활용품",
              type: "유리",
              description: "유리병",
              price: 300,
              quantity: 1,
              totalPrice: 300,
              confidence: 75
            }
          ]
        };
        setRecognizedItems(dummyResponse.mapped_items);
      } else {
        // 사용자에게 오류 알림
        Alert.alert(
          "분석 실패", 
          "이미지 분석 중 오류가 발생했습니다. 다시 시도하거나 직접 선택해주세요.",
          [{ text: "확인" }]
        );
        
        // 인식된 아이템 초기화
        setRecognizedItems([]);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 인식된 아이템 추가 함수
  const addRecognizedItems = () => {
    if (recognizedItems.length === 0) return;
    
    // 인식된 항목을 선택된 항목 목록에 추가
    setSelectedItems(prev => [...prev, ...recognizedItems]);
    
    // 총 금액 업데이트
    const additionalPrice = recognizedItems.reduce(
      (sum, item) => sum + item.totalPrice, 0
    );
    setTotalPrice(prev => prev + additionalPrice);
    
    // 추가 후 인식된 항목 목록 및 캡처된 이미지 초기화
    setRecognizedItems([]);
    setCapturedImage(null);
    
    // 사용자에게 알림
    Alert.alert("알림", "인식된 폐기물이 추가되었습니다.");
  };

  // 수량 변경 핸들러
  const handleQuantityChange = (value) => {
    setCurrentWaste(prev => ({
      ...prev,
      quantity: Math.max(1, value) // 최소 1개/kg 보장
    }));
  };

  // 폐기물 추가 핸들러
  const handleAddWaste = () => {
    if (!currentWaste.item) return;

    const newItem = {
      ...currentWaste.item,
      category: currentWaste.category,  // 카테고리 정보 추가
      quantity: currentWaste.quantity,
      totalPrice: currentWaste.item.price * currentWaste.quantity
    };

    setSelectedItems(prev => [...prev, newItem]);
    setTotalPrice(prev => prev + newItem.totalPrice);
    
    // 사용자에게 알림
    Alert.alert("알림", "폐기물이 추가되었습니다.");
  };

  // 폐기물 삭제 핸들러
  const handleRemoveWaste = (index) => {
    const removedItem = selectedItems[index];
    setSelectedItems(prev => prev.filter((_, i) => i !== index));
    setTotalPrice(prev => prev - removedItem.totalPrice);
  };

  // 캘린더 관련 함수들
  const generateCalendarDates = () => {
    const firstDay = new Date(currentYear, currentMonth, 1).getDay();
    const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();

    const dates = [];
    for (let i = 0; i < firstDay; i++) {
        dates.push(null);
    }

    for (let i = 1; i <= daysInMonth; i++) {
        dates.push(new Date(currentYear, currentMonth, i));
    }

    return dates;
  };

  // timeSlots 객체
  const timeSlots = {
    morning: [
      { display: "06:00", minutes: 360 },
      { display: "06:30", minutes: 390 },
      { display: "07:00", minutes: 420 },
      { display: "07:30", minutes: 450 },
      { display: "08:00", minutes: 480 },
      { display: "08:30", minutes: 510 },
      { display: "09:00", minutes: 540 },
      { display: "09:30", minutes: 570 },
      { display: "10:00", minutes: 600 },
      { display: "10:30", minutes: 630 },
      { display: "11:00", minutes: 660 },
      { display: "11:30", minutes: 690 }
    ],
    afternoon: [
      { display: "12:00", minutes: 720 },
      { display: "12:30", minutes: 750 },
      { display: "13:00", minutes: 780 },
      { display: "13:30", minutes: 810 },
      { display: "14:00", minutes: 840 },
      { display: "14:30", minutes: 870 },
      { display: "15:00", minutes: 900 },
      { display: "15:30", minutes: 930 },
      { display: "16:00", minutes: 960 },
      { display: "16:30", minutes: 990 },
      { display: "17:00", minutes: 1020 },
      { display: "17:30", minutes: 1050 }
    ]
  };

  const handleNextScreen = () => {
    if (currentScreen === 1) {
      if (!selectedDate || selectedTime === null) {
        Alert.alert("알림", "날짜와 시간을 선택해주세요.");
        return;
      }
    } else if (currentScreen === 2) {
      if (!userData.name || !userData.phoneNumber || !userData.roadNameAddress) {
        Alert.alert("알림", "모든 필수 정보를 입력해주세요.");
        return;
      }
    } else if (currentScreen === 3) {
      if (selectedItems.length === 0) {
        Alert.alert("알림", "최소 하나 이상의 폐기물을 선택해주세요.");
        return;
      }
      // Navigate to result page with all data
      navigation.navigate('PickupResult', {
        selectedDate,
        selectedTime,
        userData,
        selectedItems,
        totalPrice
      });
      return;
    }

    // 애니메이션 효과로 슬라이드 처리
    Animated.timing(slideAnim, {
      toValue: -width,
      duration: 300,
      useNativeDriver: true
    }).start(() => {
      setCurrentScreen(prev => prev + 1);
      slideAnim.setValue(0);
    });
  };

  const handlePrevScreen = () => {
    // 애니메이션 효과로 슬라이드 처리
    Animated.timing(slideAnim, {
      toValue: width,
      duration: 300,
      useNativeDriver: true
    }).start(() => {
      setCurrentScreen(prev => prev - 1);
      slideAnim.setValue(0);
    });
  };

  const handleInputChange = (field, value) => {
    if (field !== 'email') {
      setUserData(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const handlePrevMonth = () => {
    if (currentMonth === 0) {
      setCurrentMonth(11);
      setCurrentYear(prev => prev - 1);
    } else {
      setCurrentMonth(prev => prev - 1);
    }
  };

  const handleNextMonth = () => {
    if (currentMonth === 11) {
      setCurrentMonth(0);
      setCurrentYear(prev => prev + 1);
    } else {
      setCurrentMonth(prev => prev + 1);
    }
  };

  const isToday = (date) => {
    if (!date) return false;
    const today = new Date();
    return date.getDate() === today.getDate() &&
        date.getMonth() === today.getMonth() &&
        date.getFullYear() === today.getFullYear();
  };

  const isPastDate = (date) => {
    if (!date) return false;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return date < today;
  };

  // 시간 표시를 위한 유틸리티 함수
  const formatTimeDisplay = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;
  };

  // 인식된 폐기물 렌더링
  const renderRecognizedItems = () => {
    if (recognizedItems.length === 0) return null;
    
    return (
      <View style={extendedStyles.recognizedItemsContainer}>
        <Text style={extendedStyles.recognizedItemsHeader}>인식된 폐기물</Text>
        
        {/* 촬영한 이미지 표시 */}
        {capturedImage && (
          <View style={extendedStyles.capturedImageContainer}>
            <Image 
              source={{ uri: capturedImage }} 
              style={extendedStyles.capturedImageSmall}
              resizeMode="contain"
            />
          </View>
        )}
        
        {/* 인식된 폐기물 정보 */}
        {recognizedItems.map((item, index) => (
          <View key={index} style={extendedStyles.recognizedItem}>
            <View style={extendedStyles.recognizedItemHeader}>
              <Text style={extendedStyles.recognizedItemType}>
                {item.type} {item.description && `(${item.description})`}
              </Text>
              {item.confidence && (
                <Text style={extendedStyles.confidenceText}>정확도: {item.confidence}%</Text>
              )}
            </View>
            <View style={extendedStyles.recognizedItemDetails}>
              <Text>
                {item.quantity}
                {item.category === "재활용품" ? "kg" : "개"} x {item.price.toLocaleString()}원
              </Text>
              <Text style={extendedStyles.recognizedItemPrice}>
                {item.totalPrice.toLocaleString()}원
              </Text>
            </View>
          </View>
        ))}
        
        <TouchableOpacity
          style={extendedStyles.addRecognizedButton}
          onPress={addRecognizedItems}
        >
          <Text style={extendedStyles.addRecognizedButtonText}>인식된 폐기물 추가</Text>
        </TouchableOpacity>
      </View>
    );
  };

  // 화면 렌더링 - 캘린더
  const renderCalendarScreen = () => (
    <View style={extendedStyles.calendarSection}>
      <View style={extendedStyles.instruction}>
        <Text style={extendedStyles.iconText}>📅</Text>
        <Text style={extendedStyles.instructionText}>수거 날짜와 시간을 선택해 주세요</Text>
      </View>

      {/* 선택된 날짜와 시간 표시 섹션 */}
      {(selectedDate || selectedTime !== null) && (
        <View style={extendedStyles.selectedDatetime}>
          <Text style={extendedStyles.selectedLabel}>선택된 일시:</Text>
          <Text style={extendedStyles.selectedValue}>
            {selectedDate?.toLocaleDateString('ko-KR', {
              year: 'numeric',
              month: 'long',
              day: 'numeric'
            })}
            {selectedTime !== null ? ` ${formatTimeDisplay(selectedTime)}` : ''}
          </Text>
        </View>
      )}

      <View style={extendedStyles.calendarHeader}>
        <Text style={extendedStyles.calendarTitle}>
          {currentYear}.{currentMonth + 1}
        </Text>
        <View style={extendedStyles.arrowContainer}>
          <TouchableOpacity onPress={handlePrevMonth}>
            <Text style={extendedStyles.arrowIcon}>◀️</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={handleNextMonth}>
            <Text style={extendedStyles.arrowIcon}>▶️</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={extendedStyles.calendarGrid}>
        {["일", "월", "화", "수", "목", "금", "토"].map(day => (
          <View key={day} style={extendedStyles.calendarDayHeader}>
            <Text style={extendedStyles.dayHeaderText}>{day}</Text>
          </View>
        ))}
        {generateCalendarDates().map((date, idx) => (
          <TouchableOpacity
            key={idx}
            style={[
              extendedStyles.calendarDate,
              isToday(date) && extendedStyles.today,
              date && selectedDate && date.getDate() === selectedDate.getDate() &&
                date.getMonth() === selectedDate.getMonth() && extendedStyles.selected,
              isPastDate(date) && extendedStyles.pastDate,
              !date && extendedStyles.emptyDate
            ]}
            onPress={() => date && !isPastDate(date) && setSelectedDate(date)}
            disabled={!date || isPastDate(date)}
          >
            <Text style={[
              extendedStyles.dateText,
              isPastDate(date) && extendedStyles.pastDateText,
              date && selectedDate && date.getDate() === selectedDate.getDate() &&
                date.getMonth() === selectedDate.getMonth() && extendedStyles.selectedText
            ]}>
              {date?.getDate()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <View style={extendedStyles.timeSection}>
        <Text style={extendedStyles.timeSectionTitle}>오전</Text>
        <View style={extendedStyles.timeGrid}>
          {timeSlots.morning.map(slot => (
            <TouchableOpacity
              key={slot.display}
              style={[
                extendedStyles.timeSlot,
                selectedTime === slot.minutes && extendedStyles.selectedTime
              ]}
              onPress={() => setSelectedTime(slot.minutes)}
            >
              <Text style={[
                extendedStyles.timeText,
                selectedTime === slot.minutes && extendedStyles.selectedTimeText
              ]}>
                {slot.display}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={extendedStyles.timeSectionTitle}>오후</Text>
        <View style={extendedStyles.timeGrid}>
          {timeSlots.afternoon.map(slot => (
            <TouchableOpacity
              key={slot.display}
              style={[
                extendedStyles.timeSlot,
                selectedTime === slot.minutes && extendedStyles.selectedTime
              ]}
              onPress={() => setSelectedTime(slot.minutes)}
            >
              <Text style={[
                extendedStyles.timeText,
                selectedTime === slot.minutes && extendedStyles.selectedTimeText
              ]}>
                {slot.display}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </View>
  );

  // 화면 렌더링 - 사용자 정보 폼
  const renderFormScreen = () => (
    <View style={extendedStyles.formSection}>
      <View style={extendedStyles.instruction}>
        <View style={extendedStyles.svgIcon}>
          <Text style={extendedStyles.iconText}>📄</Text>
        </View>
        <Text style={extendedStyles.instructionText}>신청자 정보를 확인해 주세요</Text>
      </View>

      <View style={extendedStyles.formGroup}>
        <Text style={extendedStyles.formLabel}>이름</Text>
        <TextInput
          value={userData.name}
          onChangeText={(text) => handleInputChange('name', text)}
          style={extendedStyles.formInput}
          placeholder="이름을 입력해주세요."
        />
      </View>

      <View style={extendedStyles.formGroup}>
        <Text style={extendedStyles.formLabel}>연락처</Text>
        <TextInput
          value={userData.phoneNumber}
          onChangeText={(text) => handleInputChange('phoneNumber', text)}
          style={extendedStyles.formInput}
          placeholder="연락처를 입력해주세요."
          keyboardType="phone-pad"
        />
      </View>

      <View style={extendedStyles.formGroup}>
        <Text style={extendedStyles.formLabel}>이메일</Text>
        <TextInput
          value={userData.email}
          editable={false}
          style={[extendedStyles.formInput, extendedStyles.disabledInput]}
        />
      </View>

      <View style={extendedStyles.formGroup}>
        <Text style={extendedStyles.formLabel}>수거지 주소</Text>
        <View style={extendedStyles.postalCodeGroup}>
          <TextInput
            value={userData.postalCode}
            style={[extendedStyles.formInput, extendedStyles.postalCodeInput]}
            onChangeText={(text) => handleInputChange('postalCode', text)}
            placeholder="우편번호"
            editable={false}
          />
          <TouchableOpacity
            style={extendedStyles.postalCodeButton}
            onPress={() => setIsOpen(true)}
          >
            <Text style={extendedStyles.postalCodeButtonText}>주소 검색</Text>
          </TouchableOpacity>
        </View>
        <TextInput
          value={userData.roadNameAddress}
          onChangeText={(text) => handleInputChange('roadNameAddress', text)}
          style={[extendedStyles.formInput, extendedStyles.marginTop]}
          placeholder="주소를 입력해주세요."
          editable={false}
        />
        <TextInput
          value={userData.detailedAddress}
          onChangeText={(text) => handleInputChange('detailedAddress', text)}
          style={[extendedStyles.formInput, extendedStyles.marginTop, extendedStyles.textArea]}
          placeholder="상세주소와 함께 추가 요청사항을 작성해주세요."
          multiline={true}
          numberOfLines={4}
        />
      </View>
    </View>
  );

  // 화면 렌더링 - 폐기물 선택
  const renderWasteScreen = () => (
    <View style={extendedStyles.wasteSection}>
      <View style={extendedStyles.instruction}>
        <View style={extendedStyles.svgIcon}>
          <Text style={extendedStyles.iconText}>🗑️</Text>
        </View>
            <Text style={extendedStyles.instructionText}>폐기물 종류를 선택해 주세요</Text>
      </View>

      {/* 카메라로 폐기물 인식 버튼 */}
      <TouchableOpacity
        style={extendedStyles.cameraButton}
        onPress={handleOpenCamera}
      >
        <Text style={extendedStyles.cameraButtonText}>📷 사진으로 폐기물 인식</Text>
      </TouchableOpacity>

      {/* 안내 메시지 */}
      <View style={extendedStyles.guidanceContainer}>
        <Text style={extendedStyles.guidanceTitle}>⚠️ 안내사항</Text>
        <Text style={extendedStyles.guidanceText}>
          - 카메라로 폐기물을 가까이에서 찍어주세요.
          - 밝은 곳에서 촬영하면 인식률이 높아집니다.
          - 인식 결과가 부정확한 경우 직접 폐기물을 선택해주세요.
          - 여러 폐기물은 한 번에 하나씩 촬영해주세요.
        </Text>
      </View>

      {/* 분석 중 로딩 표시 */}
      {isAnalyzing && (
        <View style={extendedStyles.analyzingContainer}>
          <ActivityIndicator size="large" color="#059669" />
          <Text style={extendedStyles.analyzingText}>폐기물을 분석하고 있습니다...</Text>
        </View>
      )}

      {/* 인식된 폐기물 표시 */}
      {renderRecognizedItems()}

      {/* 폐기물 카테고리 선택 */}
      <ScrollView style={extendedStyles.wasteScrollView}>
        {Object.entries(wasteTypes).map(([category, items]) => (
          <View key={category} style={extendedStyles.wasteCategory}>
            <TouchableOpacity
              style={extendedStyles.categoryHeader}
              onPress={() => setCurrentWaste(prev => ({
                ...prev,
                category,
                item: items[0]
              }))}
            >
              <Text style={extendedStyles.categoryText}>{category}({items.length})</Text>
              <Text style={extendedStyles.arrowIcon}>▶️</Text>
            </TouchableOpacity>

            {currentWaste.category === category && (
              <View style={extendedStyles.wasteDetails}>
                <View style={extendedStyles.wasteSelectGroup}>
                  <TouchableOpacity
                    style={extendedStyles.pickerButton}
                    onPress={() => {
                      Alert.alert(
                        "폐기물 선택",
                        "폐기물을 선택해주세요",
                        items.map(item => ({
                          text: `${item.type} ${item.description || ''}`,
                          onPress: () => setCurrentWaste(prev => ({
                            ...prev,
                            item: item
                          }))
                        }))
                      );
                    }}
                  >
                    <Text style={extendedStyles.pickerButtonText}>
                      {currentWaste.item ? `${currentWaste.item.type} ${currentWaste.item.description || ''}` : '폐기물 선택'}
                    </Text>
                  </TouchableOpacity>
                </View>

                <View style={extendedStyles.quantityGroup}>
                  <View style={extendedStyles.quantityControl}>
                    <TextInput
                      keyboardType="numeric"
                      value={currentWaste.quantity.toString()}
                      onChangeText={(text) => handleQuantityChange(parseInt(text) || 1)}
                      style={extendedStyles.quantityInput}
                    />
                    <Text style={extendedStyles.quantityUnit}>
                      (단위: {currentWaste.category === "재활용품" ? "kg" : "개"})
                    </Text>
                  </View>
                  <Text style={extendedStyles.estimatedPrice}>
                    예상 금액: {currentWaste.item ? (currentWaste.item.price * currentWaste.quantity).toLocaleString() : 0}원
                  </Text>
                  <TouchableOpacity
                    style={extendedStyles.addButton}
                    onPress={handleAddWaste}
                  >
                    <Text style={extendedStyles.addButtonText}>추가하기</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>
        ))}

        {/* 선택된 폐기물 목록 */}
        <View style={extendedStyles.selectedItems}>
          <Text style={extendedStyles.selectedItemsHeader}>선택한 폐기물</Text>
          {selectedItems.length === 0 ? (
            <Text style={extendedStyles.noItems}>현재 선택한 폐기물이 없습니다.</Text>
          ) : (
            selectedItems.map((item, index) => (
              <View key={index} style={extendedStyles.selectedItem}>
                <Text style={extendedStyles.selectedItemText}>
                  {item.type} {item.description && `(${item.description})`} - {item.quantity}{item.category === "재활용품" ? "kg" : "개"} : {item.totalPrice.toLocaleString()}원
                </Text>
                <TouchableOpacity onPress={() => handleRemoveWaste(index)}>
                  <Text style={extendedStyles.deleteIcon}>🗑️</Text>
                </TouchableOpacity>
              </View>
            ))
          )}
        </View>

        {/* 총 예상 금액 */}
        <View style={extendedStyles.total}>
          <Text style={extendedStyles.totalPrice}>
            총 예상 금액: {totalPrice.toLocaleString()}원
          </Text>
          <Text style={extendedStyles.priceNote}>
            *실제 결제 금액은 예상 금액과 다를 수 있습니다.
          </Text>
        </View>
      </ScrollView>
    </View>
  );

  // 카메라 모달 렌더링
  const renderCameraModal = () => (
    <Modal
      visible={cameraVisible}
      animationType="slide"
      transparent={false}
    >
      <View style={extendedStyles.cameraContainer}>
        {hasPermission ? (
          <>
            <CameraView
              style={extendedStyles.camera}
              facing="back"
              onMountError={(error) => {
                console.error("Camera mount error:", error);
                Alert.alert('카메라 오류', '카메라를 실행하는 중 오류가 발생했습니다.');
              }}
              ref={cameraRef}
            />
            <View style={extendedStyles.cameraControls}>
              <TouchableOpacity
                style={extendedStyles.captureButton}
                onPress={handleCapture}
              >
                <Text style={extendedStyles.captureButtonText}>촬영</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={extendedStyles.closeCameraButton}
                onPress={() => setCameraVisible(false)}
              >
                <Text style={extendedStyles.closeCameraButtonText}>취소</Text>
              </TouchableOpacity>
            </View>
            
            <View style={extendedStyles.cameraInstructions}>
              <Text style={extendedStyles.cameraInstructionText}>
                폐기물을 가까이에서 촬영해주세요
              </Text>
            </View>
          </>
        ) : (
          <View style={extendedStyles.noCameraPermission}>
            <Text style={extendedStyles.noCameraPermissionText}>카메라 권한이 필요합니다</Text>
            <TouchableOpacity
              style={extendedStyles.closeButton}
              onPress={() => setCameraVisible(false)}
            >
              <Text style={extendedStyles.closeButtonText}>닫기</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </Modal>
  );

  // 디버그 모드 토글 버튼 (개발 중에만 사용, 배포 시 제거 필요)
  const renderDebugToggle = () => {
    if (__DEV__) { // React Native 개발 모드에서만 표시
      return (
        <TouchableOpacity
          style={extendedStyles.debugButton}
          onPress={() => {
            setDebugMode(!debugMode);
            Alert.alert("디버그 모드", debugMode ? "디버그 모드가 비활성화되었습니다." : "디버그 모드가 활성화되었습니다.");
          }}
        >
          <Text style={extendedStyles.debugButtonText}>
            {debugMode ? "디버그 모드 켜짐" : "디버그 모드 꺼짐"}
          </Text>
        </TouchableOpacity>
      );
    }
    return null;
  };

  return (
    <SafeAreaView style={extendedStyles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />
      <View style={extendedStyles.mainContainer}>
        {/* 디버그 모드 토글 (개발 중에만 표시) */}
        {renderDebugToggle()}

        <Animated.View style={[
          extendedStyles.content,
          {transform: [{translateX: slideAnim}]}
        ]}>
          <ScrollView contentContainerStyle={extendedStyles.scrollContent}>
            {currentScreen === 1 ? renderCalendarScreen() : 
             currentScreen === 2 ? renderFormScreen() : 
             renderWasteScreen()}
          </ScrollView>
        </Animated.View>

        <View style={extendedStyles.footerContainer}>
          {currentScreen > 1 && (
            <TouchableOpacity 
              onPress={handlePrevScreen} 
              style={extendedStyles.prevButton}
            >
              <Text style={extendedStyles.prevButtonText}>이전으로</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            onPress={handleNextScreen}
            style={[
              extendedStyles.nextButton,
              currentScreen > 1 && extendedStyles.withPrev
            ]}
          >
            <Text style={extendedStyles.nextButtonText}>다음으로 ({currentScreen}/3)</Text>
          </TouchableOpacity>
        </View>

        {/* 주소 검색 모달 */}
        <Modal
          visible={isOpen}
          animationType="slide"
          transparent={true}
        >
          <View style={extendedStyles.modalContainer}>
            <View style={extendedStyles.modalContent}>
              <TouchableOpacity
                onPress={() => setIsOpen(false)}
                style={extendedStyles.closeButton}
              >
                <Text style={extendedStyles.closeButtonText}>닫기</Text>
              </TouchableOpacity>
              <View style={extendedStyles.postcodeContainer}>
                <Postcode
                  onSelected={completeHandler}
                  style={extendedStyles.postcode}
                />
              </View>
            </View>
          </View>
        </Modal>

        {/* 카메라 모달 */}
        {renderCameraModal()}
      </View>
    </SafeAreaView>
  );
};

// 디버그 모드 스타일 추가
const debugButtonStyle = {
  position: 'absolute',
  top: 10,
  right: 10,
  backgroundColor: 'rgba(0,0,0,0.7)',
  padding: 5,
  borderRadius: 4,
  zIndex: 9999
};

const debugButtonTextStyle = {
  color: 'white',
  fontSize: 10
};

// 확장된 스타일에 디버그 스타일 추가
if (extendedStyles) {
  extendedStyles.debugButton = debugButtonStyle;
  extendedStyles.debugButtonText = debugButtonTextStyle;
}

export default PickupRequestPage;