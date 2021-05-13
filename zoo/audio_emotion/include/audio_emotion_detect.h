//
// Created by liubaojia on 20-3-21.
//

#pragma once

#include "ncnn/net.h"
#include "ncnn/cpu.h"

class AudioEmotionDetect {
public:

    AudioEmotionDetect(const char* model_path);
    AudioEmotionDetect(const char* paramPath, const char* binPath);
    ~AudioEmotionDetect();

    int inference(const unsigned char* data, const int width, const int height, bool isBgr=false);
    void release();

private:
    void initNcnnNet(const char* paramPath, const char* binPath);

    ncnn::Net models_;
};

