//
// Created by liubaojia on 20-3-21.
//

#include "audio_emotion_detect.h"

AudioEmotionDetect::AudioEmotionDetect(const char* model_path) {

    std::string path = model_path;
    initNcnnNet((path + "/audio.param").c_str(), (path + "/audio.bin").c_str());

}


AudioEmotionDetect::AudioEmotionDetect(const char* paramPath, const char* binPath){
    initNcnnNet(paramPath, binPath);
}

void AudioEmotionDetect::initNcnnNet(const char* paramPath, const char* binPath){
    if(models_.load_param(paramPath) != 0) return;
    if(models_.load_model(binPath) != 0) return;
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 2;
    models_.opt = opt;
    ncnn::set_cpu_powersave(2);
}

AudioEmotionDetect::~AudioEmotionDetect(){
    release();
}

void AudioEmotionDetect::release(){
    models_.clear();
}

int AudioEmotionDetect::inference(const unsigned char* data, const int width, const int height, bool isBgr) {
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(data, isBgr ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, width, height, 320, 64);
    ncnn::Extractor extractor = models_.create_extractor();
    ncnn::Mat output;

    const float mean[3] = {0, 0, 0};
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};

    input.substract_mean_normalize(mean, norm);
    extractor.input("image", input);
    extractor.extract("gemfield_out", output);

    float max_ratio = ((float*)(output.data))[0];
    int index=0;

    for(int j=0; j<output.cstep; j++)
    {
        const float* prob = (float*)output.data + output.c * j;
        if(prob[0] > max_ratio) {
            max_ratio = prob[0];
            index = j;
        }
    }
    return index;
}