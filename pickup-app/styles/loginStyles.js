import { StyleSheet } from 'react-native';

export default StyleSheet.create({
  loginContainer: {
    position: 'absolute', // 화면 전체를 덮도록 설정
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#1E2A38', // 배경색 적용
    justifyContent: 'center', // 중앙 정렬
    alignItems: 'center', // 중앙 정렬
  },
  loginLogo: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  loginSubtitle: {
    fontSize: 18,
    color: '#4CAF50',
    marginBottom: 20,
  },
  loginInput: {
    width: '80%',
    height: 50,
    backgroundColor: '#34495E',
    borderRadius: 8,
    paddingHorizontal: 15,
    color: '#ffffff',
    marginBottom: 15,
  },
  loginButton: {
    width: '80%',
    height: 50,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  loginButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loginFooter: {
    flexDirection: 'row',
    marginTop: 10,
  },
  loginLink: {
    color: '#4CAF50',
    marginHorizontal: 10,
  },
});
