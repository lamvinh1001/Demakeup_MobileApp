import React, { useState, useEffect, useRef } from 'react'
import { ActivityIndicator, SafeAreaView, Image, View, StyleSheet, Text, TouchableOpacity } from 'react-native'
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import Icon from 'react-native-vector-icons/FontAwesome'


function PhotoScreen({ navigation, route }) {
    const [hasPermission, setHasPermission] = useState();
    const cameraRef = useRef();
    const [type, setType] = useState(Camera.Constants.Type.back);
    const [showCamera, setShowCamera] = useState(true);
    const [base64Image, setBase64] = useState();
    const [formData, setFormData] = useState(new FormData());

    const [isLoading, setIsLoading] = useState(true)
    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            setHasPermission(status === 'granted');


            // picker permission

            if (Platform.OS !== 'web') {
                const { status } = await ImagePicker.requestCameraPermissionsAsync();
                if (status !== "granted") {
                    alert("Sorry, we need camera roll permission to make this work!")
                }
            }
        })();
    }, []);
    if (hasPermission === null) {
        return <View />;
    }
    if (hasPermission === false) {
        return <Text>No access to camera</Text>;
    }

    const takePhoto = async () => {
        if (cameraRef) {
            try {
                let photo = await cameraRef.current.takePictureAsync({
                    allowsEditing: true,
                    aspect: [4, 3],
                    quality: 1,
                    base64: true,
                });
                return photo;
            } catch (e) {
                console.log(e);
            }
        }
    };
    const pickImage = async () => {
        // No permissions request is necessary for launching the image library
        let rs = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [5, 6],
            quality: 1,
            base64: true,
        });
        if (!rs.cancelled) {
            var formPicker = new FormData()

            setBase64(rs.base64);
            formPicker.append('base64', rs.base64)
            formPicker.append('uri', rs.uri)
            setFormData(formPicker)
        }
    };

    const generate = async () => {
        // http://13.250.97.192:5000/post

        const res = await fetch("http://ec2-18-136-212-145.ap-southeast-1.compute.amazonaws.com:8000/post", {
            // url 
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'multipart/form-data',
            },
            body: formData

        }).then(res => { console.log("Sent") })
            .catch(res => { console.log('Error') })




    }

    if (showCamera) {
        return (
            <View style={{ flex: 1 }}>
                <SafeAreaView style={{ flex: 0.8, marginTop: 35 }}>
                    <Camera style={{ flex: 1 }} type={type} ref={cameraRef}>
                    </Camera>
                </SafeAreaView>
                <View style={styles.buttonContainer}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={() => {
                            setType(
                                type === Camera.Constants.Type.back
                                    ? Camera.Constants.Type.front
                                    : Camera.Constants.Type.back
                            );
                        }}>
                        <View style={styles.buttonIcon}>
                            <Icon name="undo" color="#eee" size={30} />
                        </View>
                        <Text style={styles.buttonText}> Flip </Text>
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.button}
                        onPress={async () => {
                            const r = await takePhoto();
                            setShowCamera(false)
                            if (!r.cancelled) {
                                var formPhoto = new FormData();
                                // navigation.navigate('GeneratorScreen', {
                                //     paramKey: r.base64,
                                // })
                                setBase64(r.base64)
                                formPhoto.append('base64', r.base64)
                                formPhoto.append('uri', r.uri)
                                setFormData(formPhoto)
                            }
                        }}
                    >
                        <View style={styles.buttonIcon}>
                            <Icon name="camera" color="#eee" size={30} />
                        </View>
                        <Text style={styles.buttonText}> Photo </Text>
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.button}
                        onPress={async () => {
                            setShowCamera(false)
                            await pickImage();
                        }}
                    >
                        <View style={styles.buttonIcon}>
                            <Icon name="photo" color="#eee" size={30} />
                        </View>
                        <Text style={styles.buttonText}> Gallery </Text>
                    </TouchableOpacity>


                    {/* <TouchableOpacity style={styles.button}
                        onPress={() => navigation.goBack()}
                    >
                        <View>
                            <Icon name="close" color="#eee" size={38} />
                        </View>
                        <Text style={styles.buttonText}> Cancel </Text>
                    </TouchableOpacity> */}
                </View>
            </View>
        )
    }
    else {
        if (!isLoading) {
            return (
                <View style={{ justifyContent: 'center', backgroundColor: '#d3d3d3', textAlign: 'center', height: '100%', width: '100%' }}>
                    <ActivityIndicator
                        color='black'
                        size='large'
                        animated={false}
                    />
                    <Text style={{ textAlign: 'center', fontSize: 18, fontWeight: 'bold', marginTop: 5 }}>Is in the process of generating</Text>
                </View>
            )
        }
        else {
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
                        > Input </Text>

                        <View>
                            {base64Image ?
                                <Image
                                    style={styles.imageField}

                                    source={{ uri: 'data:image/png;base64,' + base64Image }}
                                />
                                : <Image
                                    style={styles.imageField}

                                    source={require('../assets/unknow.png')}
                                />}

                        </View>
                    </View>


                    <View style={{ flexDirection: 'row', }}>
                        <TouchableOpacity
                            style={styles.buttonTitle}
                            onPress={async () =>
                                setShowCamera(true)
                            }
                        >
                            <Text
                                style={styles.buttonText}
                            > Picture Again</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.buttonTitle}
                            onPress={async () => {
                                setIsLoading(false);
                                var start = new Date().getTime();
                                await generate();
                                // http://13.250.97.192:5000/predict

                                const rs = await fetch("http://ec2-18-136-212-145.ap-southeast-1.compute.amazonaws.com:8000/predict", {
                                    method: 'GET',
                                    headers: {
                                        'Accept': 'application/string',
                                        'Content-Type': 'application/string',
                                    },
                                }).then((response) => {
                                    return response.text();
                                }).then((data) => {

                                    navigation.push('GeneratorScreen', {
                                        post: data,
                                        timeGenerator: Math.abs(start - new Date().getTime()) / 1000,
                                    }
                                    )
                                }).finally(() => setIsLoading(true))



                            }}
                        >
                            <Text
                                style={styles.buttonText}
                            > Generate Face</Text>
                        </TouchableOpacity>
                    </View>


                </View>);
        }


    }

}
export const styles = StyleSheet.create({
    buttonContainer: {
        flex: 0.2,
        backgroundColor: "transparent",
        flexDirection: "row",

        backgroundColor: "#000"
    },

    button: {
        flex: 1,
        alignSelf: "flex-end",
        alignItems: "center",
        marginBottom: 50
    },
    buttonText: {
        color: "#fff",
        fontWeight: "bold",
        fontSize: 15,
    },
    buttonIcon: {
        marginBottom: 5,
    },
    imageField: {
        marginBottom: 50,
        width: 410,
        height: 410,
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

});

export default PhotoScreen;