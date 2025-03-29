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

// --- Helper function to handle writing a processed chunk ---
static void write_audio_chunk(
    // Non-owning references/pointers to the necessary context:
    std::vector<std::unique_ptr<StreamingWavWriter>>& writers,
    std::vector<float>& interleaved_write_buffer,
    // The actual chunk data passed by the callback mechanism:
    const Eigen::Tensor<float, 3>& chunk_data,
    int num_valid_samples) // Keep int here as it represents a count, less likely to exceed INT_MAX
{
    if (num_valid_samples <= 0) return; // Nothing to write

    // Use auto or Eigen::Index to match the return type of dimension() exactly
    const auto current_num_sources = chunk_data.dimension(0);
    const auto num_channels = chunk_data.dimension(1); // Should be 2

    // Ensure num_channels is reasonable before proceeding (optional sanity check)
    if (num_channels != 2) {
        std::cerr << "Warning: static_write_audio_chunk expected 2 channels, got " << num_channels << std::endl;
        // Decide how to handle this - maybe return or proceed cautiously
    }

    // Ensure the buffer is large enough
    // Use size_t for sizes and capacities
    const size_t required_buffer_size = static_cast<size_t>(num_valid_samples) * static_cast<size_t>(num_channels);
    if (interleaved_write_buffer.size() < required_buffer_size) {
        interleaved_write_buffer.resize(required_buffer_size);
    }

    // --- Loop using size_t for index 'i' ---
    // Cast current_num_sources to size_t for the loop limit comparison
    const size_t num_sources_t = static_cast<size_t>(current_num_sources);

    for (size_t i = 0; i < num_sources_t; ++i) {
        // Check writer index against writers.size() (both are size_t now)
        if (i >= writers.size() || !writers[i]) {
            std::cerr << "Warning: Invalid writer for source index " << i << std::endl;
            continue;
        }

        // Interleave the data for the current source (i)
        for (int k = 0; k < num_valid_samples; ++k) {
            // Access using Eigen::Index types implicitly converted from size_t/int
            // Or explicitly cast if needed, but usually okay:
            // long long source_idx = static_cast<long long>(i); // If Eigen::Index is long long
            interleaved_write_buffer[static_cast<size_t>(k) * static_cast<size_t>(num_channels) + 0] = chunk_data(i, 0, k); // Left channel
            interleaved_write_buffer[static_cast<size_t>(k) * static_cast<size_t>(num_channels) + 1] = chunk_data(i, 1, k); // Right channel
        }

        // Write the interleaved data using the appropriate writer
        // Pass the required_buffer_size which is already size_t
        writers[i]->append_samples(interleaved_write_buffer.data(), required_buffer_size);
    }
}

// --- Main Incremental Processing Function ---

extern "C" int process_demucs_onnx(int argc, const char **argv) {
    try {
        // --- Argument Parsing ---
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <model file> <wav file> <out dir>" << std::endl;
            return 1;
        }
        std::cout << "demucs.onnx Incremental Main driver program" << std::endl;
        std::string model_file = argv[1];
        std::string wav_file = argv[2];
        std::string out_dir = argv[3];
        std::filesystem::path output_dir_path(out_dir);

        // --- Directory Handling ---
        if (!std::filesystem::exists(output_dir_path)) {
            std::cerr << "Directory does not exist: " << out_dir << ". Creating it." << std::endl;
            if (!std::filesystem::create_directories(output_dir_path)) {
                std::cerr << "Error: Unable to create directory: " << out_dir << std::endl;
                return 1;
            }
        } else if (!std::filesystem::is_directory(output_dir_path)) {
            std::cerr << "Error: " << out_dir << " exists but is not a directory!" << std::endl;
            return 1;
        }

        // --- Load Audio ---
        std::cout << "Loading audio file: " << wav_file << std::endl;
        Eigen::MatrixXf audio = load_audio_file(wav_file);

        // --- Load Model & Setup ONNX Session ---
        std::cout << "Loading model file: " << model_file << std::endl;
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern();
        session_options.DisableCpuMemArena();
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        // --- Reverted thread settings to original hardcoded values ---
        session_options.SetIntraOpNumThreads(3);
        session_options.SetInterOpNumThreads(3);
        // --- End of reverted settings ---
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Add any platform-specific providers if needed (e.g., CoreML, NNAPI)
        // session_options.AppendExecutionProvider_CoreML(COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);

        demucsonnx::demucs_model model = load_model(model_file, session_options);
        std::cout << "Model loaded successfully (" << model.nb_sources << " sources)." << std::endl;

        // --- Prepare Streaming Writers ---
        int nb_out_sources = model.nb_sources;
        std::vector<std::unique_ptr<StreamingWavWriter>> writers;
        std::vector<std::string> target_names; // Store names for logging/finalization messages

        for (int target = 0; target < nb_out_sources; ++target) {
            std::string target_name;
             switch (target) { // Assuming standard Demucs v3/v4 names + common extras
                 case 0: target_name = "drums"; break;
                 case 1: target_name = "bass"; break;
                 case 2: target_name = "other"; break;
                 case 3: target_name = "vocals"; break;
                 case 4: target_name = "guitar"; break;
                 case 5: target_name = "piano"; break;
                 default:
                     std::cerr << "Warning: Target index " << target << " not recognized, using generic name 'source_" << target << "'." << std::endl;
                     target_name = "source_" + std::to_string(target);
             }
             target_names.push_back(target_name); // Store the name

            std::filesystem::path p_target = output_dir_path / (target_name + ".wav");
            std::cout << "Preparing output file: " << p_target.string() << std::endl;

            auto writer = std::make_unique<StreamingWavWriter>();
            if (!writer->open(p_target.string(), demucsonnx::SUPPORTED_SAMPLE_RATE, 2)) { // Always open as stereo
                std::cerr << "Error: Failed to open writer for " << target_name << std::endl;
                // Clean up already opened writers? Maybe not necessary if exiting.
                return 1; // Critical error
            }
            writers.push_back(std::move(writer));
        }

        // --- Create the reusable buffer for interleaving ---
        std::vector<float> interleaved_write_buffer; // Exists in this scope

        // --- Create the Callback Function Object using std::bind ---
        using namespace std::placeholders; // For _1, _2

        demucsonnx::WriteChunkCallback write_chunk_callback_obj =
            std::bind(&write_audio_chunk,
                      std::ref(writers),                  // Pass writers vector by reference
                      std::ref(interleaved_write_buffer), // Pass buffer by reference
                      _1,                                 // Placeholder for chunk_data
                      _2);                                // Placeholder for num_valid_samples


        // --- Define Progress Callback ---
        std::cout << "Running Incremental Demucs.onnx inference for: " << wav_file << std::endl;
        std::cout << std::fixed << std::setprecision(1); // Progress formatting
        demucsonnx::ProgressCallback progressCallback = [&](float progress, const std::string &log_message) {
             // Ensure progress stays within [0, 100] range visually
             float display_progress = std::max(0.0f, std::min(100.0f, progress * 100.0f));
             std::cout << "[" << std::setw(5) << display_progress << "%] " << log_message << std::endl;
        };


        // --- Run Incremental Inference with the Bound Callback ---
        demucsonnx::demucs_inference_incremental(model, audio, progressCallback, write_chunk_callback_obj);


        // --- Finalize Writers ---
        std::cout << "Finalizing output files..." << std::endl;
        bool all_finalized = true;
        for (size_t i = 0; i < writers.size(); ++i) {
             // Check writer exists before finalizing (should always exist here unless setup failed)
             if (writers[i] && !writers[i]->finalize()) {
                 // Use the stored name for better error message
                 std::cerr << "Error: Failed to finalize writer for " << (i < target_names.size() ? target_names[i] : "unknown source") << std::endl;
                 all_finalized = false;
                 // Continue finalizing others even if one fails
             }
        }

        if (!all_finalized) {
            std::cerr << "Warning: One or more output files may not have been finalized correctly." << std::endl;
            // Decide if this should be a fatal error (return 1) depending on requirements
        }

        std::cout << "Incremental processing complete." << std::endl;
        return 0; // Success

    } catch (const std::exception &e) {
        std::cerr << "Error: Exception caught during processing: " << e.what() << std::endl;
        return 1; // Indicate failure
    } catch (...) {
        // Catch any other non-standard exceptions
        std::cerr << "Error: Unknown exception caught during processing." << std::endl;
        return 1; // Indicate failure
    }
}
