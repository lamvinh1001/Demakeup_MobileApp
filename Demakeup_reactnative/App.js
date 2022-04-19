
import React from 'react'

import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import GeneratorScreen from './components/generate'
import HomeScreen from './components/home'
import PhotoScreen from './components/photo'

const Stack = createNativeStackNavigator();

function App() {
  return (

    <NavigationContainer >
      <Stack.Navigator initialRouteName='HomeScreen' screenOptions={{ headerShown: false }}>
        <Stack.Screen name="HomeScreen" component={HomeScreen} />
        <Stack.Screen name="PhotoScreen" component={PhotoScreen} />
        <Stack.Screen name="GeneratorScreen" component={GeneratorScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;






