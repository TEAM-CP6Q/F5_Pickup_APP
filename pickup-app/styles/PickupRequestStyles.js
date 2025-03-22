import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

const PickupRequestStyles = StyleSheet.create({
    mainContainer: {
        flex: 1,
        backgroundColor: "#fff",
    },
    content: {
        flex: 1,
    },
    scrollContent: {
        flexGrow: 1,
        paddingHorizontal: 16,
        paddingBottom: 16,
    },
    instruction: {
        flexDirection: "row",
        alignItems: "center",
        marginVertical: 16,
    },
    instructionText: {
        marginLeft: 8,
        fontSize: 16,
        fontWeight: "600",
    },
    svgIcon: {
        width: 24,
        height: 24,
        justifyContent: 'center',
        alignItems: 'center',
    },
    iconText: {
        fontSize: 20,
    },
    
    // 캘린더 스타일
    calendarSection: {
        flex: 1,
    },
    selectedDatetime: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 16,
        backgroundColor: "#f0f9ff",
        padding: 10,
        borderRadius: 5,
    },
    selectedLabel: {
        fontWeight: "600",
        marginRight: 8,
    },
    selectedValue: {
        flex: 1,
    },
    calendarHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 16,
    },
    calendarTitle: {
        fontSize: 18,
        fontWeight: "600",
    },
    arrowContainer: {
        flexDirection: "row",
        gap: 8,
    },
    arrowIcon: {
        fontSize: 20,
    },
    deleteIcon: {
        fontSize: 22,
        color: "#ff4d4d",
    },
    calendarGrid: {
        flexDirection: "row",
        flexWrap: "wrap",
        borderWidth: 1,
        borderColor: "#e0e0e0",
        marginBottom: 20,
    },
    calendarDayHeader: {
        width: `${100/7}%`,
        paddingVertical: 10,
        alignItems: "center",
        backgroundColor: "#f5f5f5",
    },
    dayHeaderText: {
        fontWeight: "600",
    },
    calendarDate: {
        width: `${100/7}%`,
        height: 45,
        justifyContent: "center",
        alignItems: "center",
        borderWidth: 0.5,
        borderColor: "#e0e0e0",
    },
    dateText: {
        color: "#333",
    },
    today: {
        backgroundColor: "#f0f9ff",
    },
    selected: {
        backgroundColor: "#059669",
    },
    selectedText: {
        color: "#fff",
        fontWeight: "bold",
    },
    pastDate: {
        backgroundColor: "#f5f5f5",
    },
    pastDateText: {
        color: "#aaa",
    },
    emptyDate: {
        backgroundColor: "#fafafa",
    },
    timeSection: {
        marginTop: 16,
    },
    timeSectionTitle: {
        fontSize: 16,
        fontWeight: "600",
        marginBottom: 8,
    },
    timeGrid: {
        flexDirection: "row",
        flexWrap: "wrap",
        marginBottom: 16,
    },
    timeSlot: {
        width: `${100/4 - 2}%`,
        marginHorizontal: "1%",
        marginVertical: 5,
        paddingVertical: 12,
        alignItems: "center",
        borderWidth: 1,
        borderColor: "#e0e0e0",
        borderRadius: 5,
    },
    timeText: {
        color: "#333",
    },
    selectedTime: {
        backgroundColor: "#059669",
        borderColor: "#059669",
    },
    selectedTimeText: {
        color: "#fff",
    },
    
    // 폼 스타일
    formSection: {
        flex: 1,
    },
    formGroup: {
        marginBottom: 16,
    },
    formLabel: {
        fontSize: 14,
        marginBottom: 8,
        fontWeight: "500",
    },
    formInput: {
        borderWidth: 1,
        borderColor: "#e0e0e0",
        borderRadius: 8,
        padding: 12,
        backgroundColor: "#fff",
    },
    marginTop: {
        marginTop: 8,
    },
    textArea: {
        height: 100,
        textAlignVertical: "top",
    },
    disabledInput: {
        backgroundColor: "#f5f5f5",
    },
    postalCodeGroup: {
        flexDirection: "row",
        gap: 8,
    },
    postalCodeInput: {
        flex: 1,
    },
    postalCodeButton: {
        backgroundColor: "#059669",
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 8,
        justifyContent: "center",
    },
    postalCodeButtonText: {
        color: "#fff",
        fontWeight: "500",
    },
    
    // 폐기물 선택 스타일
    wasteSection: {
        flex: 1,
    },
    wasteScrollView: {
        flex: 1,
    },
    wasteCategory: {
        marginBottom: 16,
        borderWidth: 1,
        borderColor: "#e5e7eb",
        borderRadius: 12,
        backgroundColor: "#fff",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 4,
        elevation: 2,
    },
    categoryHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        padding: 16,
    },
    categoryText: {
        fontSize: 16,
        fontWeight: "500",
        color: "#1f2937",
    },
    wasteDetails: {
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: "#e5e7eb",
        backgroundColor: "#f8fafc",
        borderBottomLeftRadius: 12,
        borderBottomRightRadius: 12,
    },
    wasteSelectGroup: {
        marginBottom: 16,
    },
    pickerButton: {
        backgroundColor: "#fff",
        borderWidth: 1.5,
        borderColor: "#e5e7eb",
        borderRadius: 8,
        padding: 12,
    },
    pickerButtonText: {
        color: "#1f2937",
        fontSize: 16,
    },
    quantityGroup: {
        backgroundColor: "#fff",
        padding: 16,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: "#e5e7eb",
    },
    quantityControl: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 16,
    },
    quantityInput: {
        width: 100,
        padding: 10,
        borderWidth: 1.5,
        borderColor: "#e5e7eb",
        borderRadius: 6,
        textAlign: "center",
    },
    quantityUnit: {
        color: "#6b7280",
        fontSize: 14,
        marginLeft: 8,
    },
    estimatedPrice: {
        fontSize: 16,
        color: "#059669",
        fontWeight: "500",
        marginBottom: 16,
        textAlign: "right",
    },
    addButton: {
        backgroundColor: "#059669",
        padding: 12,
        borderRadius: 6,
        alignItems: "center",
    },
    addButtonText: {
        color: "#fff",
        fontWeight: "500",
        fontSize: 16,
    },
    
    // 선택된 폐기물 목록
    selectedItems: {
        marginTop: 32,
        backgroundColor: "#fff",
        borderRadius: 12,
        padding: 16,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 4,
        elevation: 2,
        maxHeight: 300,
    },
    selectedItemsHeader: {
        fontSize: 18,
        fontWeight: "600",
        color: "#1f2937",
        marginBottom: 16,
    },
    selectedItem: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        padding: 16,
        backgroundColor: "#f8fafc",
        borderWidth: 1,
        borderColor: "#e5e7eb",
        borderRadius: 8,
        marginBottom: 12,
    },
    selectedItemText: {
        fontSize: 15,
        color: "#1f2937",
        flex: 1,
        marginRight: 8,
    },
    noItems: {
        textAlign: "center",
        color: "#6b7280",
        padding: 16,
        borderWidth: 1,
        borderColor: "#e5e7eb",
        borderRadius: 8,
    },
    total: {
        marginTop: 32,
        paddingTop: 16,
        borderTopWidth: 1,
        borderTopColor: "#e5e7eb",
    },
    totalPrice: {
        textAlign: "right",
        fontWeight: "600",
        fontSize: 16,
    },
    priceNote: {
        fontSize: 14,
        color: "#6b7280",
        fontWeight: "normal",
        marginTop: 4,
        textAlign: "right",
    },
    
    // 푸터 스타일
    footerContainer: {
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: "#e5e7eb",
        flexDirection: "row",
        gap: 16,
    },
    nextButton: {
        flex: 1,
        padding: 16,
        backgroundColor: "#fff",
        borderWidth: 1,
        borderColor: "#059669",
        borderRadius: 8,
        alignItems: "center",
    },
    nextButtonText: {
        color: "#059669",
        fontWeight: "500",
    },
    withPrev: {
        flex: 0.5,
    },
    prevButton: {
        flex: 0.5,
        padding: 16,
        backgroundColor: "#fff",
        borderWidth: 1,
        borderColor: "#059669",
        borderRadius: 8,
        alignItems: "center",
    },
    prevButtonText: {
        color: "#059669",
        fontWeight: "500",
    },
    
    // 모달 스타일
    modalContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "rgba(0, 0, 0, 0.5)",
    },
    modalContent: {
        width: "90%",
        height: 500,
        backgroundColor: "#fff",
        borderRadius: 8,
        overflow: "hidden",
    },
    closeButton: {
        alignSelf: "center",
        padding: 12,
        margin: 8,
    },
    closeButtonText: {
        fontSize: 16,
        color: "#000",
    },
    postcodeContainer: {
        flex: 1,
        width: "100%",
    },
    postcode: {
        width: "100%",
        height: "100%",
    },
    
    // 카메라 관련 스타일
    cameraButton: {
        backgroundColor: "#059669",
        padding: 16,
        borderRadius: 8,
        alignItems: "center",
        marginBottom: 16,
    },
    cameraButtonText: {
        color: "#fff",
        fontWeight: "500",
        fontSize: 16,
    },
    cameraContainer: {
        flex: 1,
        backgroundColor: "#000",
    },
    camera: {
        flex: 1,
        backgroundColor: "#000",
    },
    cameraPlaceholder: {
        textAlign: 'center',
        fontSize: 18,
        color: '#fff',
        marginTop: '50%'
    },
    cameraControls: {
        position: "absolute",
        bottom: 30,
        left: 0,
        right: 0,
        flexDirection: "row",
        justifyContent: "center",
        padding: 20,
    },
    captureButton: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: "#fff",
        justifyContent: "center",
        alignItems: "center",
    },
    captureButtonText: {
        fontSize: 16,
        fontWeight: "600",
    },
    previewContainer: {
        flex: 1,
        backgroundColor: "#000",
    },
    previewImage: {
        flex: 1,
        resizeMode: "contain",
    },
    previewControls: {
        position: "absolute",
        bottom: 30,
        left: 0,
        right: 0,
        flexDirection: "row",
        justifyContent: "space-around",
        padding: 20,
    },
    previewButton: {
        paddingVertical: 10,
        paddingHorizontal: 20,
        backgroundColor: "#059669",
        borderRadius: 8,
    },
    previewButtonText: {
        color: "#fff",
        fontSize: 16,
        fontWeight: "600",
    },
    retakeButton: {
        backgroundColor: "#6b7280",
    },
    noCameraPermission: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#fff",
    },
    noCameraPermissionText: {
        fontSize: 16,
        marginBottom: 20,
    },
    capturedImageContainer: {
        height: 150,
        borderRadius: 8,
        overflow: 'hidden',
        marginBottom: 16,
    },
    capturedImage: {
        width: '100%',
        height: '100%',
    },
    analyzingContainer: {
        padding: 20,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: "#f5f5f5",
        borderRadius: 8,
        marginBottom: 16,
    },
    analyzingText: {
        marginTop: 10,
        fontSize: 16,
        color: "#333",
    },
    recognizedItemsContainer: {
        backgroundColor: "#ecfdf5",
        borderWidth: 1,
        borderColor: "#059669",
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
    },
    recognizedItemsHeader: {
        fontSize: 18,
        fontWeight: "600",
        color: "#059669",
        marginBottom: 12,
    },
    recognizedItem: {
        backgroundColor: "#fff",
        padding: 12,
        borderRadius: 6,
        marginBottom: 8,
    },
    recognizedItemText: {
        fontSize: 14,
        color: "#333",
    },
    addRecognizedButton: {
        backgroundColor: "#059669",
        padding: 12,
        borderRadius: 6,
        alignItems: "center",
        marginTop: 8,
    },
    addRecognizedButtonText: {
        color: "#fff",
        fontWeight: "500",
    }
,
// styles.js 파일에 추가할 스타일
confidenceText: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic'
},

// 추가 도움말 스타일
wasteHelpText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginVertical: 10,
    fontStyle: 'italic'
},

// 이미지 인식 상태 표시
recognitionIndicator: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: '#059669',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 10
},
recognitionText: {
    color: 'white',
    fontSize: 12
},
// 카메라 관련 스타일
cameraContainer: {
    flex: 1,
    backgroundColor: '#000'
  },
  camera: {
    flex: 1
  },
  cameraControls: {
    position: 'absolute',
    bottom: 30,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center'
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 20,
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.5)'
  },
  captureButtonText: {
    fontSize: 16,
    color: '#059669',
    fontWeight: 'bold'
  },
  closeCameraButton: {
    position: 'absolute',
    right: 20,
    padding: 10
  },
  closeCameraButtonText: {
    color: 'white',
    fontSize: 16
  },
  cameraInstructions: {
    position: 'absolute',
    top: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.4)'
  },
  cameraInstructionText: {
    color: 'white',
    fontSize: 14
  },
  noCameraPermission: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f1f1f1'
  },
  noCameraPermissionText: {
    fontSize: 16,
    marginBottom: 20
  },
  
  // 인식된 폐기물 관련 스타일
  recognizedItemsContainer: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 16,
    marginTop: 10,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#e5e7eb'
  },
  recognizedItemsHeader: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#059669'
  },
  capturedImageContainer: {
    width: '100%',
    marginBottom: 15,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#e5e7eb'
  },
  capturedImageSmall: {
    width: '100%',
    height: 120,
    borderRadius: 8
  },
  recognizedItem: {
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 6,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb'
  },
  recognizedItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8
  },
  recognizedItemType: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937'
  },
  recognizedItemDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 5,
    paddingTop: 5,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb'
  },
  recognizedItemPrice: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#059669'
  },
  confidenceText: {
    fontSize: 12,
    color: '#6b7280',
    fontStyle: 'italic'
  },
  addRecognizedButton: {
    backgroundColor: '#059669',
    borderRadius: 6,
    padding: 12,
    alignItems: 'center',
    marginTop: 12
  },
  addRecognizedButtonText: {
    color: 'white',
    fontWeight: 'bold'
  },
  
  // 분석 중 표시
  analyzingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: 8,
    marginVertical: 15
  },
  analyzingText: {
    marginTop: 10,
    fontSize: 14,
    color: '#4b5563'
  },
  
  // 안내 메시지 스타일
  guidanceContainer: {
    backgroundColor: '#f0f9ff',
    borderRadius: 8,
    padding: 12,
    marginVertical: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#0284c7'
  },
  guidanceTitle: {
    fontWeight: 'bold',
    fontSize: 14,
    marginBottom: 5,
    color: '#0284c7'
  },
  guidanceText: {
    fontSize: 12,
    lineHeight: 18,
    color: '#334155'
  },
  
  // 기타 스타일
  cameraButton: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 10,
    borderWidth: 1,
    borderColor: '#d1d5db'
  },
  cameraButtonText: {
    fontSize: 16,
    color: '#1f2937',
    marginLeft: 8
  },
});


export default PickupRequestStyles;