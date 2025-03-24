#include "demucs.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stddef.h>
#include <tuple>
#include <vector>
using namespace nqr;

static demucsonnx::demucs_model load_model(const std::string& htdemucs_model_path, Ort::SessionOptions& session_options) {
    struct demucsonnx::demucs_model model;
    std::ifstream file(htdemucs_model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + htdemucs_model_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> file_data(size);
    if (!file.read(file_data.data(), size)) {
        throw std::runtime_error("Failed to read model file.");
    }
    bool success = demucsonnx::load_model(file_data, model, session_options);
    if (!success) {
        throw std::runtime_error("Failed to load model.");
    }
    return model;
}

static Eigen::MatrixXf load_audio_file(std::string filename)
{
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    NyquistIO loader;
    loader.Load(fileData.get(), filename);
    if (fileData->sampleRate != demucsonnx::SUPPORTED_SAMPLE_RATE)
    {
        std::cerr << "[ERROR] demucs.cpp only supports the following sample rate (Hz): " << demucsonnx::SUPPORTED_SAMPLE_RATE << std::endl;
        throw std::runtime_error("Unsupported sample rate");
    }
    std::cout << "Input samples: " << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;
    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr << "[ERROR] demucs.cpp only supports mono and stereo audio" << std::endl;
        throw std::runtime_error("Unsupported channel count");
    }
    std::size_t N = fileData->samples.size() / fileData->channelCount;
    Eigen::MatrixXf ret(2, N);
    if (fileData->channelCount == 1)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[i];
            ret(1, i) = fileData->samples[i];
        }
    }
    else
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[2 * i];
            ret(1, i) = fileData->samples[2 * i + 1];
        }
    }
    return ret;
}

static void write_audio_file(const Eigen::MatrixXf &waveform, std::string filename)
{
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    fileData->sampleRate = demucsonnx::SUPPORTED_SAMPLE_RATE;
    fileData->channelCount = 2;
    fileData->samples.resize(waveform.cols() * 2);
    for (long int i = 0; i < waveform.cols(); ++i)
    {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }
    int encoderStatus = encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE}, fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}

enum DemucsError {
    SUCCESS = 0,
    INVALID_ARGUMENTS = 1,
    DIRECTORY_ERROR = 2,
    AUDIO_LOAD_ERROR = 3,
    MODEL_LOAD_ERROR = 4,
    INFERENCE_ERROR = 5,
    WRITE_ERROR = 6,
    UNSUPPORTED_TARGET = 7
};

int run_demucs_onnx(std::string model_file_path, std::string wav_file_path, std::string out_dir)
{
    std::cout << "demucs.onnx Main driver program" << std::endl;
    std::filesystem::path output_dir_path(out_dir);
    if (!std::filesystem::exists(output_dir_path))
    {
        std::cerr << "Directory does not exist: " << out_dir << ". Creating it." << std::endl;
        if (!std::filesystem::create_directories(output_dir_path))
        {
            std::cerr << "Error: Unable to create directory: " << out_dir << std::endl;
            return DIRECTORY_ERROR;
        }
    }
    else if (!std::filesystem::is_directory(output_dir_path))
    {
        std::cerr << "Error: " << out_dir << " exists but is not a directory!" << std::endl;
        return DIRECTORY_ERROR;
    }
    Eigen::MatrixXf audio;
    try {
        audio = load_audio_file(wav_file_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading audio file: " << e.what() << std::endl;
        return AUDIO_LOAD_ERROR;
    }
    Eigen::Tensor3dXf out_targets;
    std::cout << "Running Demucs.onnx inference for: " << wav_file_path << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    demucsonnx::ProgressCallback progressCallback = [](float progress, const std::string &log_message)
    {
        std::cout << "(" << std::setw(3) << std::setfill(' ') << progress * 100.0f << "%) " << log_message << std::endl;
    };
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetIntraOpNumThreads(16);
    session_options.SetInterOpNumThreads(16);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    struct demucsonnx::demucs_model model;
    try {
        model = load_model(model_file_path, session_options);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return MODEL_LOAD_ERROR;
    }
    Eigen::Tensor3dXf audio_targets;
    try {
        audio_targets = demucsonnx::demucs_inference(model, audio, progressCallback);
        out_targets = audio_targets;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return INFERENCE_ERROR;
    }
    int nb_out_sources = model.nb_sources;
    for (int target = 0; target < nb_out_sources; ++target)
    {
        std::filesystem::path p = out_dir;
        std::filesystem::create_directories(p);
        auto p_target = p / "target_0.wav";
        std::string target_name;
        switch (target)
        {
        case 0:
            target_name = "drums";
            break;
        case 1:
            target_name = "bass";
            break;
        case 2:
            target_name = "other";
            break;
        case 3:
            target_name = "vocals";
            break;
        case 4:
            target_name = "guitar";
            break;
        case 5:
            target_name = "piano";
            break;
        default:
            std::cerr << "Error: target " << target << " not supported" << std::endl;
            return UNSUPPORTED_TARGET;
        }
        p_target.replace_filename("target_" + std::to_string(target) + "_" + target_name + ".wav");
        std::cout << "Writing wav file " << p_target << std::endl;
        Eigen::MatrixXf target_waveform(2, audio.cols());
        for (int channel = 0; channel < 2; ++channel)
        {
            for (int sample = 0; sample < audio.cols(); ++sample)
            {
                target_waveform(channel, sample) = out_targets(target, channel, sample);
            }
        }
        try {
            write_audio_file(target_waveform, p_target.string());
        } catch (const std::exception& e) {
            std::cerr << "Error writing audio file: " << e.what() << std::endl;
            return WRITE_ERROR;
        }
    }
    return SUCCESS;
}
