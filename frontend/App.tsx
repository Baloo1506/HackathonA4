import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import PredictionScreen from './src/PredictionScreen';

export default function App() {
  return (
    <SafeAreaView style={s.root}>
      <PredictionScreen />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0d14' },
});


