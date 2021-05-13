#pragma once

#include <vector>
#include <unordered_map>

#include "ncnn/net.h"
#include "ncnn/cpu.h"

class SceneClassify {
public:

    SceneClassify(const char* model_path);
    ~SceneClassify();
    std::vector<int> inference(const unsigned char* data, const int width, const int height, bool isBgr=false);
    void releaseRes();

private:
    ncnn::Net models_;
};


class Strategy {
public:
    static std::vector<std::vector<int>> frameExtract(const int* frameIndex, const int index, const int rate) noexcept {
        if(!index) return {};
        if(index == 1) return {{frameIndex[0]}};
        std::vector<std::vector<int>> all_frames;
        const int dummy_frame = 5;
        for(auto i = 0; i < index-1; ++i) {
            float jump = (frameIndex[i+1] - frameIndex[i] - dummy_frame*2) / float(rate);

            float start = frameIndex[i] + dummy_frame;
            int end = frameIndex[i+1] - dummy_frame;

            if(jump <= 0) {
                start = frameIndex[i];
                end = frameIndex[i] + 1;
                jump = 1;
            } else if(jump > 0 && jump < 1) {
                jump = 1;
            }

            std::vector<int> frames;
            while(start < end) {
                frames.push_back(start);
                start += jump;
            }
            all_frames.push_back(frames);
        }

        return all_frames;
    }

    static std::vector<std::vector<int>> frameExtractV1(const int* frameIndex, const int index, const int frameRate) noexcept {
        if(!index) return {};
        if(index == 1) return {{frameIndex[0]}};
        std::vector<std::vector<int>> all_frames;
        //抛弃每个镜头初始的N帧和最后的N帧
        const int dummy_frame = 5;
        for(auto i = 0; i < index-1; ++i) {
            int rate = (frameIndex[i+1] - frameIndex[i] - dummy_frame*2) * 2.0 / frameRate;
            float jump = (frameIndex[i+1] - frameIndex[i] - dummy_frame*2) / float(rate);

            float start = frameIndex[i] + dummy_frame;
            int end = frameIndex[i+1] - dummy_frame;

            if(jump <= 0) {
                start = frameIndex[i];
                end = frameIndex[i] + 1;
                jump = 1;
            } else if(jump > 0 && jump < 1) {
                jump = 1;
            }

            std::vector<int> frames;
            while(start < end) {
                frames.push_back(start);
                start += jump;
            }
            all_frames.push_back(frames);
        }

        return all_frames;
    }


    static std::vector<int> findMaxValue(const std::vector<std::vector<int>>& indexes) noexcept {
        std::vector<int> result;

        for(auto& vec : indexes) {
            result.push_back(Strategy::mostFrequent(vec));
        }

        return result;
    }

    static int mostFrequent(const std::vector<int>& arr) noexcept {
        std::unordered_map<int, int> hash;
        for (auto n : arr) {
            hash[n]++;
        }

        int max_count = 0, res = -1;
        for (auto i : hash) {
            if (max_count < i.second) {
                res = i.first;
                max_count = i.second;
            }
        }

        return res;
    }
};
