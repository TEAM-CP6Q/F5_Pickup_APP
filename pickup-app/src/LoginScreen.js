import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, TouchableWithoutFeedback, Keyboard, Alert } from 'react-native';
import styles from '../styles/loginStyles'; // 스타일 파일 import
import { useNavigation } from '@react-navigation/native'; // 네비게이션 사용
import AsyncStorage from '@react-native-async-storage/async-storage'; // AsyncStorage import

const LoginScreen = () => {
  const navigation = useNavigation(); // 네비게이션 훅 사용
  const [email, setEmail] = useState(''); // 이메일 상태
  const [password, setPassword] = useState(''); // 비밀번호 상태

  const handleLogin = async () => {
    // 이메일과 비밀번호가 비어있는지 확인
    if (!email || !password) {
      Alert.alert('입력 오류', '이메일과 비밀번호를 입력해주세요.');
      return;
    }

    try {
      const response = await fetch('https://refresh-f5-server.o-r.kr/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email,
          password: password,
        }),
      });

      const result = await response.json();

      if (response.status === 200) {
        // 로그인 성공
        console.log(result);
        await AsyncStorage.setItem('token', result.token); // 토큰 저장
        await AsyncStorage.setItem('email', result.user.email); // 이메일 저장
        await AsyncStorage.setItem('id', result.user.id.toString()); // 사용자 ID 저장
        await AsyncStorage.setItem('role', result.user.role); // 사용자 역할 저장
        console.log('로그인 성공');

        // 홈 화면으로 이동
        navigation.navigate('pickupMain');
      } else {
        // 로그인 실패
        Alert.alert('로그인 실패', result.message || '로그인에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (error) {
      // 서버 통신 실패
      console.error(error);
      Alert.alert('오류', '서버와의 연결에 실패했습니다. 다시 시도해주세요.');
    }
  };

  return (
    <TouchableWithoutFeedback onPress={() => Keyboard.dismiss()}>
      <View style={styles.loginContainer}>
        <Text style={styles.loginLogo}>REFRESH</Text>
        <Text style={styles.loginSubtitle}>DRIVER</Text>
        <TextInput
          placeholder="E-mail"
          style={styles.loginInput}
          placeholderTextColor="#FFFFFF"
          value={email}
          onChangeText={setEmail} // 이메일 입력값 업데이트
        />
        <TextInput
          placeholder="비밀번호"
          style={styles.loginInput}
          secureTextEntry
          placeholderTextColor="#FFFFFF"
          value={password}
          onChangeText={setPassword} // 비밀번호 입력값 업데이트
        />
        <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
          <Text style={styles.loginButtonText}>로그인</Text>
        </TouchableOpacity>
        <View style={styles.loginFooter}>
          <Text style={styles.loginLink}>비밀번호를 잊어버렸습니까?</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Signup')}>
            <Text style={styles.loginLink}>계정 생성</Text>
          </TouchableOpacity>
        </View>
      </View>
    </TouchableWithoutFeedback>
  );
};

export default LoginScreen;
