#include "SceneClassify.h"

#include <sys/time.h>



//#include <android/log.h>
//#define  LOG_TAG    "Test"
//#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


SceneClassify::SceneClassify(const char* model_path) {
    std::string path = model_path;

    int s1 = models_.load_param((path + "/sls.param").c_str());
    int s2 = models_.load_model((path + "/sls.bin").c_str());

    const char* m = std::to_string(s1).c_str();
    const char* m1 = std::to_string(s2).c_str();

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 2;
    models_.opt = opt;

    ncnn::set_cpu_powersave(2);
}

SceneClassify::~SceneClassify(){
    releaseRes();
}

void SceneClassify::releaseRes() {
    models_.clear();
}

std::vector<int> SceneClassify::inference(const unsigned char* data, const int width, const int height, bool isBgr) {
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(data, isBgr ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB, width, height, 224, 224);

    ncnn::Extractor extractor = models_.create_extractor();
    ncnn::Mat output;

    const float mean[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

    input.substract_mean_normalize(mean, norm);

    struct timeval start;
    struct timeval end;
    unsigned long timer;
    gettimeofday(&start, NULL);
    extractor.input("input", input);
    extractor.extract("output", output);
    gettimeofday(&end, NULL);
    timer = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000;

    float max_ratio = ((float*)(output.data))[0];
    int index=0;
    std::string test = "";

    for(int j=0; j<output.cstep; j++)
    {
        const float* prob = (float*)output.data + output.c * j;
        test += std::to_string(prob[0]) + ", ";
        if(prob[0] > max_ratio) {
            max_ratio = prob[0];
            index = j;
        }
    }
    test += " index:"+std::to_string(index);

//    const char* m = test.c_str();
//    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "%s", m);

    std::vector<int> infos;
    infos.push_back(timer);
    infos.push_back(index);
    infos.push_back((int)(max_ratio*100)); // score // 扩大100倍
    return infos;
}