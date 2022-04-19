import React from 'react'


import { SafeAreaView, Image, View, Text, TouchableOpacity } from 'react-native'
import { AntDesign } from '@expo/vector-icons';



function HomeScreen({ navigation }) {

    return (
        <SafeAreaView
            style={{
                flex: 1,
                backgroundColor: '#ffffff'
            }}
        >
            <View style={{
                flex: 0.7, justifyContent: "center",
                alignItems: "center",
            }} >
                <Image
                    style={{
                        marginTop: 50,
                        width: 300,
                        height: 300,
                    }}
                    source={
                        require('../assets/logo.png')
                    }
                />
                <Text style={{ color: 'black', fontWeight: 'bold', fontSize: 20, marginTop: 50 }}> Welcome to De-Makeup App!! </Text>

            </View>

            <View style={{ flex: 0.3, width: '100%', justifyContent: 'center', alignItems: 'center', marginBottom: 20, }}>
                <TouchableOpacity

                    onPress={() =>
                        navigation.navigate('PhotoScreen',

                        )}
                    // style={{ width: '60%', height: '20%', borderColor: 'black', borderWidth: 1, borderRadius: 50, backgroundColor: 'green', }}
                    style={{ backgroundColor: 'green', flexDirection: 'row', justifyContent: 'center', alignItems: "center", width: '50%', height: '20%', borderRadius: 50 }}
                >
                    <View >
                        <Text style={{ color: 'white', fontWeight: 'bold', fontSize: 20 }}> Get Started

                        </Text>
                    </View>
                    <View style={{ marginTop: 3, marginLeft: 10 }}>
                        <AntDesign name="play" size={25} color="white" />
                    </View>



                </TouchableOpacity>
            </View>
        </SafeAreaView >
    );
}
export default HomeScreen
