import React, { useEffect } from 'react'
import { View, StyleSheet, Text, TouchableOpacity, Image } from 'react-native'


function GeneratorScreen({ navigation, route }) {
    // console.log(route.params.post)
    useEffect(() => {
        if (route.params?.post) {
            // Post updated, do something with `route.params.post`
            // For example, send the post to the server
        }
    }, [route.params?.post]);
    return (

        <View
            style={{
                flex: 1,
                justifyContent: "center",
                alignItems: "center",
            }}

        >

            <View
                style={{

                    justifyContent: "center",
                    alignItems: "center",
                }}
            >
                <Text
                    style={{ fontWeight: "bold", fontSize: 20, marginBottom: 50 }}
                > Output </Text>
                <Image
                    style={styles.imageField}
                    // source={require('../assets/unknow.png')}
                    source={{ uri: 'data:image/png;base64,' + route.params?.post }}
                />
            </View>

            <View style={{ flexDirection: 'row', }}>
                <TouchableOpacity
                    style={styles.buttonTitle}
                    onPress={() =>
                        navigation.navigate('PhotoScreen',)
                    }
                >
                    <Text
                        style={{ fontSize: 15, color: '#fff', fontWeight: 'bold' }}
                    > Try Again</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.buttonTitle}
                    onPress={() =>
                        navigation.popToTop()
                    }
                >

                    <Text
                        style={{ fontSize: 15, color: '#fff', fontWeight: 'bold' }}
                    > Quit</Text>
                </TouchableOpacity>
            </View>


        </View >


    );
}
export const styles = StyleSheet.create({

    imageField: {
        marginBottom: 50,
        width: 410,
        height: 410,
    },
    buttonText: {
        color: "#fff",
        fontWeight: "bold",
        fontSize: 15,
    },
    buttonTitle: {
        borderColor: "black",
        borderStyle: "solid",
        borderWidth: 1,
        fontWeight: "bold",
        justifyContent: 'center',
        alignItems: "center",
        width: '40%',
        height: '30%',
        marginTop: 20,
        marginLeft: 10,
        borderRadius: 100,
        backgroundColor: 'green'
    },
})
export default GeneratorScreen;