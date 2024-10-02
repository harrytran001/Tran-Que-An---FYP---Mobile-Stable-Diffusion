#pragma once
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
#include "cpu.h"
using namespace std;

class DecodeSlover
{
public:
    int load(AAssetManager* mgr);

    ncnn::Mat decode(ncnn::Mat sample);

private:

    const float factor[4] = { 1.0 / 0.18215f, 1.0 / 0.18215f, 1.0 / 0.18215f, 1.0 / 0.18215f };

    const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
    const float _norm_[3] = { 127.5f, 127.5f, 127.5f };

    ncnn::Net net;
};