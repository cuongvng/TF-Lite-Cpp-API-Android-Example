#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Example: load a tflite model using TF Lite C++ API
// Credit to https://github.com/ValYouW/crossplatform-tflite-object-detecion
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_tflitecxx_MainActivity_loadModelJNI(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager,
        jstring fileName
        ) {

    char* buffer = nullptr;
    long size = 0;
    const char* modelpath = env->GetStringUTFChars(fileName, 0);

    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromBuffer(buffer, size);
    assert(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    assert(interpreter != nullptr);

    // Allocate tensor buffers.
    assert(interpreter->AllocateTensors() == kTfLiteOk);

    std::string status = "Load TF Lite model successfully!";
    return env->NewStringUTF(status.c_str());
}