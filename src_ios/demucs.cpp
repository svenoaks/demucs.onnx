#include "demucs.hpp"
#include "wav_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <map>
#include <memory>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stddef.h>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

using namespace nqr;

static demucsonnx::demucs_model load_model(const std::string& htdemucs_model_path, Ort::SessionOptions& session_options) {
    demucsonnx::demucs_model model;
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

static Eigen::MatrixXf load_audio_file(std::string filename) {
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    NyquistIO loader;
    loader.Load(fileData.get(), filename);
    if (fileData->sampleRate != demucsonnx::SUPPORTED_SAMPLE_RATE) {
        std::cerr << "[ERROR] demucs.cpp only supports the following sample rate (Hz): " << demucsonnx::SUPPORTED_SAMPLE_RATE << std::endl;
        exit(1);
    }
    std::cout << "Input samples: " << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;
    if (fileData->channelCount != 2 && fileData->channelCount != 1) {
        std::cerr << "[ERROR] demucs.cpp only supports mono and stereo audio" << std::endl;
        exit(1);
    }
    std::size_t N = fileData->samples.size() / fileData->channelCount;
    Eigen::MatrixXf ret(2, N);
    if (fileData->channelCount == 1) {
        for (std::size_t i = 0; i < N; ++i) {
            ret(0, i) = fileData->samples[i];
            ret(1, i) = fileData->samples[i];
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            ret(0, i) = fileData->samples[2 * i];
            ret(1, i) = fileData->samples[2 * i + 1];
        }
    }
    return ret;
}

static void write_audio_file(const Eigen::MatrixXf &waveform, std::string filename) {
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    fileData->sampleRate = demucsonnx::SUPPORTED_SAMPLE_RATE;
    fileData->channelCount = 2;
    fileData->samples.resize(waveform.cols() * 2);
    for (long int i = 0; i < waveform.cols(); ++i) {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }
    int encoderStatus = encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE}, fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}

extern "C" int process_demucs_onnx(int argc, const char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <model file> <wav file> <out dir>" << std::endl;
            return 1;
        }
        std::cout << "demucs.onnx Incremental Main driver program" << std::endl;
        std::string model_file = argv[1];
        std::string wav_file = argv[2];
        std::string out_dir = argv[3];
        std::filesystem::path output_dir_path(out_dir);

        // --- Directory Handling (same as before) ---
        if (!std::filesystem::exists(output_dir_path)) {
            std::cerr << "Directory does not exist: " << out_dir << ". Creating it." << std::endl;
            if (!std::filesystem::create_directories(output_dir_path)) {
                std::cerr << "Error: Unable to create directory: " << out_dir << std::endl; return 1;
            }
        } else if (!std::filesystem::is_directory(output_dir_path)) {
            std::cerr << "Error: " << out_dir << " exists but is not a directory!" << std::endl; return 1;
        }

        // --- Load Audio (same as before) ---
        Eigen::MatrixXf audio = load_audio_file(wav_file);

        // --- Load Model (same as before) ---
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern();
        session_options.DisableCpuMemArena();
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.SetIntraOpNumThreads(4); // Adjust threads as needed
        session_options.SetInterOpNumThreads(4); // Adjust threads as needed
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        demucsonnx::demucs_model model = load_model(model_file, session_options);

        // --- Prepare Streaming Writers ---
        int nb_out_sources = model.nb_sources;
        std::vector<std::unique_ptr<StreamingWavWriter>> writers;
        std::vector<std::string> target_names; // Store names for logging

        for (int target = 0; target < nb_out_sources; ++target) {
            std::string target_name;
             switch (target) {
                 case 0: target_name = "drums"; break;
                 case 1: target_name = "bass"; break;
                 case 2: target_name = "other"; break;
                 case 3: target_name = "vocals"; break;
                 case 4: target_name = "guitar"; break; // Assuming htdemucs
                 case 5: target_name = "piano"; break;  // Assuming htdemucs
                 default:
                     std::cerr << "Warning: Target index " << target << " not recognized, using generic name." << std::endl;
                     target_name = "source_" + std::to_string(target);
             }
             target_names.push_back(target_name);

            std::filesystem::path p_target = output_dir_path / (target_name + ".wav");
            std::cout << "Preparing output file: " << p_target.string() << std::endl;

            auto writer = std::make_unique<StreamingWavWriter>();
            if (!writer->open(p_target.string(), demucsonnx::SUPPORTED_SAMPLE_RATE, 2)) { // Assuming stereo output
                std::cerr << "Error: Failed to open writer for " << target_name << std::endl;
                return 1; // Critical error
            }
            writers.push_back(std::move(writer));
        }

        // --- Run Incremental Inference and Writing ---
        std::cout << "Running Incremental Demucs.onnx inference for: " << wav_file << std::endl;
        std::cout << std::fixed << std::setprecision(1); // Adjust precision for progress
        demucsonnx::ProgressCallback progressCallback = [&](float progress, const std::string &log_message) {
            // More detailed progress possible here if the incremental function provides it
             std::cout << "[" << std::setw(5) << progress * 100.0f << "%] " << log_message << std::endl;
        };

        // Call the new incremental function from model_apply.cpp
        demucsonnx::model_inference_and_write_incremental(model, audio, writers, progressCallback);


        // --- Finalize Writers (updates headers, closes files) ---
        std::cout << "Finalizing output files..." << std::endl;
        bool all_finalized = true;
        for (size_t i = 0; i < writers.size(); ++i) {
            if (!writers[i]->finalize()) {
                std::cerr << "Error: Failed to finalize writer for " << target_names[i] << std::endl;
                all_finalized = false;
                // Continue finalizing others
            }
        }

        if (!all_finalized) {
            std::cerr << "Warning: One or more output files may be corrupted." << std::endl;
            // Decide if this should be a fatal error (return 1)
        }

        std::cout << "Incremental processing complete." << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught." << std::endl;
        return 1;
    }
}


