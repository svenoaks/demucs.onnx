// demucs.cpp
// Consolidated file containing C++ implementation of C API wrapper and static helpers.
// Calls core logic defined elsewhere (e.g., model_apply.cpp).

// --- Standard C++ and Library Includes ---
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <filesystem> // Requires C++17
#include <iostream>
#include <iomanip>
#include <memory>     // For std::unique_ptr
#include <functional> // For std::function, std::bind, std::ref, std::placeholders
#include <stdexcept>
#include <thread>     // For std::thread::hardware_concurrency
#include <atomic>     // For std::atomic<bool>
#include <cmath>      // For std::max, std::min, std::ceil, std::sqrt
#include <random>     // For random shift generation
#include <fstream>    // For ifstream used in load_model

// --- Project-Specific Includes ---
#include "demucs.hpp"       // Core C++ declarations (demucs_model, callbacks, constants, load_model declaration)
#include "demucs_result_codes.h" // Shared enum definition
#include "wav_writer.hpp"   // For StreamingWavWriter class
#include <libnyquist/Decoders.h> // For loading audio via libnyquist
#include <libnyquist/Common.h>   // For nqr::AudioData
#include <onnxruntime/core/session/onnxruntime_cxx_api.h> // For ONNX Runtime
#include <unsupported/Eigen/CXX11/Tensor> // For Eigen::Tensor (used by write_audio_chunk)

// --- Include C Interface Header ---
#include "demucs_interface.h"

// --- Forward Declaration for the Core Inference Function ---
// This function MUST be defined in a linked file (e.g., model_apply.cpp).

DemucsResultCode demucsonnx::demucs_inference_incremental(
                                                          struct demucsonnx::demucs_model &model,
                                                          const Eigen::MatrixXf &audio,
                                                          demucsonnx::ProgressCallback cb,
                                                          const WriteChunkCallback& write_callback,
                                                          const std::atomic<bool>* cancel_flag);



// --- Static Helper Function Definitions (Internal to demucs.cpp) ---

// --- Audio loading ---
static Eigen::MatrixXf load_audio_file_internal(const std::string& filename) {
    std::shared_ptr<nqr::AudioData> fileData = std::make_shared<nqr::AudioData>();
    nqr::NyquistIO loader;
    if (!std::filesystem::exists(filename)) { throw std::runtime_error("[load_audio_internal] Input audio file does not exist: " + filename); }
    try { loader.Load(fileData.get(), filename); }
    catch (const std::exception& e) { throw std::runtime_error("[load_audio_internal] libnyquist load failed: " + std::string(e.what())); }
    catch (...) { throw std::runtime_error("[load_audio_internal] libnyquist load failed with unknown exception."); } // Catch non-std exceptions
    if (!fileData || fileData->samples.empty()) { std::cerr << "Warning [load_audio_internal]: Audio data empty: " << filename << std::endl; return Eigen::MatrixXf(2, 0); }
    if (fileData->sampleRate != demucsonnx::SUPPORTED_SAMPLE_RATE) { throw std::runtime_error("[load_audio_internal] Unsupported sample rate..."); }
    if (fileData->channelCount != 1 && fileData->channelCount != 2) { throw std::runtime_error("[load_audio_internal] Unsupported channel count..."); }
    std::size_t N = fileData->samples.size() / fileData->channelCount; Eigen::MatrixXf ret(2, N);
    if (fileData->channelCount == 1) { for (std::size_t i = 0; i < N; ++i) { ret(0, i) = fileData->samples[i]; ret(1, i) = fileData->samples[i]; } }
    else { for (std::size_t i = 0; i < N; ++i) { ret(0, i) = fileData->samples[2 * i]; ret(1, i) = fileData->samples[2 * i + 1]; } }
    std::cout << "Info [load_audio_internal]: Loaded " << N << " frames, " << fileData->channelCount << " channels from " << filename << std::endl;
    return ret;
}

// --- Model loading ---
static demucsonnx::demucs_model load_model_internal(const std::string& htdemucs_model_path, Ort::SessionOptions& session_options) {
    demucsonnx::demucs_model model; std::ifstream file; std::vector<char> file_data;
    try { // Wrap file operations
        file.open(htdemucs_model_path, std::ios::binary | std::ios::ate);
        if (!file) { throw std::runtime_error("Failed to open model file stream"); }
        std::streamsize size = file.tellg();
        if (size <= 0) { throw std::runtime_error("Model file empty or size error"); }
        file.seekg(0, std::ios::beg); file_data.resize(size);
        if (!file.read(file_data.data(), size)) { throw std::runtime_error("Failed to read model file to buffer"); }
        file.close();
    } catch (const std::exception& e) {
        throw std::runtime_error("[load_model_internal] File IO error: " + std::string(e.what()) + " (" + htdemucs_model_path + ")");
    } catch (...) {
         if(file.is_open()) file.close();
         throw std::runtime_error("[load_model_internal] Unknown file IO error (" + htdemucs_model_path + ")");
    }

    bool success = false;
    try { // Wrap demucsonnx::load_model call
        success = demucsonnx::load_model(file_data, model, session_options); // Assumes this is defined elsewhere
    } catch (const std::exception& e) {
         throw std::runtime_error("[load_model_internal] demucsonnx::load_model failed: " + std::string(e.what()));
    } catch (...) {
         throw std::runtime_error("[load_model_internal] demucsonnx::load_model failed with unknown exception.");
    }

    if (!success) { throw std::runtime_error("[load_model_internal] demucsonnx::load_model returned false."); }
    if (model.nb_sources <= 0) { std::cerr << "Warning [load_model_internal]: Model loaded with " << model.nb_sources << " sources." << std::endl; }
    std::cout << "Info [load_model_internal]: Model loaded with " << model.nb_sources << " sources." << std::endl;
    return model;
}

// --- Static writer helper ---
// Takes the output writers, a reusable buffer, and the processed data chunk
static void write_audio_chunk_internal(
    std::vector<std::unique_ptr<StreamingWavWriter>>& writers, // Vector of writers for each source
    std::vector<float>& interleaved_write_buffer,           // Reusable buffer for interleaving
    const Eigen::Tensor<float, 3>& chunk_data,              // Processed data (source, channel, sample)
    int num_valid_samples)                                  // Number of samples in this chunk
{
    if (num_valid_samples <= 0) {
        // std::cout << "Debug [write_audio_chunk_internal]: num_valid_samples is 0, skipping." << std::endl; // Optional debug log
        return; // Nothing to write
    }

    // Get dimensions from the tensor
    const auto current_num_sources = chunk_data.dimension(0);
    const auto num_channels = chunk_data.dimension(1);
    const auto chunk_num_samples_dim = chunk_data.dimension(2); // Actual sample dimension size in tensor

    // Validate channel count
    if (num_channels != 2) {
         std::cerr << "Error [write_audio_chunk_internal]: Expected 2 channels in chunk data, got " << num_channels << std::endl;
         return; // Cannot proceed with incorrect channel count
    }

    // Ensure num_valid_samples doesn't exceed the tensor's dimension
    if (num_valid_samples > chunk_num_samples_dim) {
        std::cerr << "Warning [write_audio_chunk_internal]: num_valid_samples (" << num_valid_samples
                  << ") exceeds chunk data dimension (" << chunk_num_samples_dim << "). Clamping." << std::endl;
        num_valid_samples = chunk_num_samples_dim;
        if (num_valid_samples <= 0) return; // Return if clamping results in zero samples
    }

    // Ensure the reusable buffer is large enough for interleaving
    const size_t required_buffer_size = static_cast<size_t>(num_valid_samples) * static_cast<size_t>(num_channels);
    try { // Protect against potential bad_alloc during resize
        if (interleaved_write_buffer.size() < required_buffer_size) {
            interleaved_write_buffer.resize(required_buffer_size);
        }
    } catch (const std::exception& e) {
         std::cerr << "Error [write_audio_chunk_internal]: Failed to resize write buffer: " << e.what() << std::endl;
         return; // Cannot proceed without buffer
    }


    // Loop through each output source provided in chunk_data
    for (size_t i = 0; static_cast<Eigen::Index>(i) < current_num_sources; ++i) {
        // Check if a writer exists and is valid for this source index
        if (i >= writers.size() || !writers[i]) {
            std::cerr << "Warning [write_audio_chunk_internal]: Invalid or missing writer for source index " << i << std::endl;
            continue; // Skip this source
        }

        // --- Interleave the stereo data for the current source (i) ---
        bool bounds_error_occurred = false;
        for (int k = 0; k < num_valid_samples; ++k) {
            // Calculate index in the 1D interleaved buffer
            size_t base_idx = static_cast<size_t>(k) * static_cast<size_t>(num_channels);

            // Basic bounds check before accessing Tensor (k is already checked against num_valid_samples which is clamped)
            // Check source index 'i' and channel indices 0, 1
             if (static_cast<Eigen::Index>(i) >= chunk_data.dimension(0) || 1 >= chunk_data.dimension(1)) {
                 std::cerr << "Error [write_audio_chunk_internal]: Source/Channel index out of bounds during interleave (i=" << i << ", k=" << k << ")." << std::endl;
                 // Fill remaining part of buffer with zeros for this source and stop processing it
                 for (size_t fill_idx = base_idx; fill_idx < required_buffer_size; ++fill_idx) {
                     if(fill_idx < interleaved_write_buffer.size()) interleaved_write_buffer[fill_idx] = 0.0f;
                 }
                 bounds_error_occurred = true;
                 break; // Stop processing samples for this source
            }

            // Access Tensor elements using (source, channel, sample) indices
            // Left channel
            interleaved_write_buffer[base_idx + 0] = chunk_data(i, 0, k);
            // Right channel
            interleaved_write_buffer[base_idx + 1] = chunk_data(i, 1, k);
        } // End loop over samples (k)

        // If a bounds error occurred while interleaving, skip writing for this source
        if (bounds_error_occurred) {
            continue;
        }

        // --- Write the interleaved data using the appropriate StreamingWavWriter ---
        // Pass the total number of float samples in the buffer for this chunk
        if (!writers[i]->append_samples(interleaved_write_buffer.data(), required_buffer_size)) {
             // append_samples should ideally log its own errors, but we can log here too
             std::cerr << "Error [write_audio_chunk_internal]: writer->append_samples failed for source index " << i << std::endl;
             // Decide how to handle this - continue processing other sources? Abort entirely?
             // For now, just log and continue.
        }
    } // End loop over sources (i)
} // End write_audio_chunk_internal

// --- Implementation of the public C API function ---
extern "C" DemucsResultCode process_demucs_onnx_c(
    const char* model_path_c,
    const char* input_wav_path_c,
    const char* output_dir_path_c,
    DemucsProgressCallback_C progress_callback_c,
    void* progress_callback_context,
    void* cancel_flag_context)
{
    const std::atomic<bool>* cancel_flag_atomic_ptr = static_cast<const std::atomic<bool>*>(cancel_flag_context);

    if (!model_path_c || !input_wav_path_c || !output_dir_path_c || !cancel_flag_atomic_ptr) {
        std::cerr << "Error [C API]: Null path or cancel flag context provided." << std::endl;
        return DEMUCS_RESULT_ERROR_INVALID_ARGS;
    }
    if (cancel_flag_atomic_ptr->load(std::memory_order_relaxed)) {
        std::cerr << "Info [C API]: Processing cancelled at entry." << std::endl;
        return DEMUCS_RESULT_CANCELLED;
    }

    // Declare variables needed across try blocks or for finalization
    std::vector<std::unique_ptr<StreamingWavWriter>> writers;
    std::vector<std::string> target_names;
    DemucsResultCode final_result = DEMUCS_RESULT_SUCCESS; // Assume success initially

    try { // Outer try for setup stages before main inference call
        std::string model_file = model_path_c;
        std::string wav_file = input_wav_path_c;
        std::string out_dir = output_dir_path_c;
        std::filesystem::path output_dir_path(out_dir);

        // Directory Handling
        std::error_code fs_error_code;
        if (!std::filesystem::exists(output_dir_path)) {
            if (!std::filesystem::create_directories(output_dir_path, fs_error_code) || fs_error_code) {
                 std::cerr << "Error [C API]: Unable to create output directory: " << out_dir << " (" << fs_error_code.message() << ")" << std::endl;
                 return DEMUCS_RESULT_ERROR_OUTPUT_DIR; // Early exit on setup failure
            }
        } else if (!std::filesystem::is_directory(output_dir_path)) {
             std::cerr << "Error [C API]: Output path exists but is not a directory: " << out_dir << std::endl;
             return DEMUCS_RESULT_ERROR_OUTPUT_DIR; // Early exit
        }

        // Load Audio
        Eigen::MatrixXf audio_data;
        try { audio_data = load_audio_file_internal(wav_file); }
        catch (const std::exception& e) { std::cerr << "Error [C API]: Failed loading audio '" << wav_file << "': " << e.what() << std::endl; return DEMUCS_RESULT_ERROR_AUDIO_LOAD; }
        catch (...) { std::cerr << "Error [C API]: Unknown error loading audio '" << wav_file << "'." << std::endl; return DEMUCS_RESULT_ERROR_AUDIO_LOAD; }
        if (audio_data.cols() == 0) { std::cerr << "Info [C API]: Audio file loaded successfully but contains no samples: " << wav_file << std::endl; return DEMUCS_RESULT_SUCCESS; }

        // Load Model & Setup Session
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern(); session_options.DisableCpuMemArena(); session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        session_options.SetIntraOpNumThreads(3); session_options.SetInterOpNumThreads(3);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        demucsonnx::demucs_model model;
        try { model = load_model_internal(model_file, session_options); }
        catch (const std::exception& e) { std::cerr << "Error [C API]: Failed model load '" << model_file << "': " << e.what() << std::endl; return DEMUCS_RESULT_ERROR_MODEL_LOAD; }
        catch (...) { std::cerr << "Error [C API]: Unknown error loading model '" << model_file << "'." << std::endl; return DEMUCS_RESULT_ERROR_MODEL_LOAD; }
        if (model.nb_sources <= 0) { std::cerr << "Error [C API]: Model loaded with invalid number of sources: " << model.nb_sources << std::endl; return DEMUCS_RESULT_ERROR_MODEL_LOAD; }

        // Prepare Streaming Writers (with added exception safety)
        int nb_out_sources = model.nb_sources;
        // Clear vectors in case of retry logic (though not present here)
        writers.clear();
        target_names.clear();
        try { // Wrap the loop that might throw
            for (int target = 0; target < nb_out_sources; ++target) {
                std::string target_name;
                switch (target) {
                     case 0: target_name = "drums"; break; case 1: target_name = "bass"; break;
                     case 2: target_name = "other"; break; case 3: target_name = "vocals"; break;
                     case 4: target_name = "guitar"; break; case 5: target_name = "piano"; break;
                     default: target_name = "source_" + std::to_string(target); break;
                 }
                 if (target_name.empty()) { std::cerr << "Error [C API]: Generated empty target name for index " << target << std::endl; return DEMUCS_RESULT_ERROR_UNKNOWN; }
                 target_names.push_back(target_name);
                 std::string filename_only = target_name + ".wav";
                 std::filesystem::path p_target = output_dir_path / filename_only;
                 std::cout << "Info [C API]: Preparing writer for: " << p_target.string() << std::endl;
                 auto writer = std::make_unique<StreamingWavWriter>(); // Can throw bad_alloc
                 if (!writer->open(p_target.string(), demucsonnx::SUPPORTED_SAMPLE_RATE, 2)) { // open might throw or return false
                     std::cerr << "Error [C API]: Failed opening writer: " << p_target.string() << std::endl;
                     return DEMUCS_RESULT_ERROR_WRITER_OPEN; // Return specific error
                 }
                 writers.push_back(std::move(writer));
            }
        } catch (const std::exception& e) {
             std::cerr << "Error [C API]: Exception during writer preparation: " << e.what() << std::endl;
             return DEMUCS_RESULT_ERROR_WRITER_OPEN; // Treat setup issues as writer open error
        } catch (...) {
             std::cerr << "Error [C API]: Unknown exception during writer preparation." << std::endl;
             return DEMUCS_RESULT_ERROR_WRITER_OPEN;
        }


        // Prepare C++ Callbacks (with added exception safety)
        std::vector<float> interleaved_write_buffer;
        demucsonnx::WriteChunkCallback write_chunk_callback_obj;
        demucsonnx::ProgressCallback progressCallbackInternal;
        try { // Wrap potentially throwing std::bind/lambda creation
            using namespace std::placeholders;
            write_chunk_callback_obj =
                std::bind(&write_audio_chunk_internal, std::ref(writers), std::ref(interleaved_write_buffer), _1, _2);

            if (progress_callback_c) {
                progressCallbackInternal = [=](float p, const std::string& m){ if (!cancel_flag_atomic_ptr->load()) progress_callback_c(progress_callback_context, p, m.c_str()); };
            } else { progressCallbackInternal = [](float, const std::string&){}; }
        } catch (const std::exception& e) {
             std::cerr << "Error [C API]: Exception during callback preparation: " << e.what() << std::endl;
             return DEMUCS_RESULT_ERROR_UNKNOWN; // Or CPP_EXCEPTION
        } catch (...) {
             std::cerr << "Error [C API]: Unknown exception during callback preparation." << std::endl;
             return DEMUCS_RESULT_ERROR_UNKNOWN;
        }


        // Run In-Memory Incremental Inference (call external function)
        std::cout << "Info [C API]: Starting inference call..." << std::endl;
        try {
            final_result = demucsonnx::demucs_inference_incremental(
                model, audio_data, progressCallbackInternal, write_chunk_callback_obj, cancel_flag_atomic_ptr
            );
        } catch (const std::exception& e) {
            final_result = DEMUCS_RESULT_ERROR_INFERENCE; // Assign specific code
            std::cerr << "Error [C API]: std::exception caught during inference call: " << e.what() << std::endl;
        } catch (...) {
            final_result = DEMUCS_RESULT_ERROR_INFERENCE; // Assign specific code
            std::cerr << "Error [C API]: Unknown exception caught during inference call." << std::endl;
        }
        std::cout << "Info [C API]: Inference call returned with code: " << final_result << std::endl;

        // audio_data goes out of scope here

    // Catch C++ Exceptions from the setup stages (audio/model load, directory, writer/callback prep)
    } catch (const std::exception &e) {
        std::cerr << "Error [C API]: C++ std::exception caught during setup: " << e.what() << std::endl;
        // Determine more specific error code based on where it likely happened?
        // For now, use a general C++ exception code.
        final_result = DEMUCS_RESULT_ERROR_CPP_EXCEPTION;
    } catch (...) {
        std::cerr << "Error [C API]: Unknown C++ exception caught during setup." << std::endl;
        final_result = DEMUCS_RESULT_ERROR_UNKNOWN;
    }

    // --- Finalize Writers (Always attempt unless setup failed badly before writers were created) ---
    // We attempt finalization even if inference failed or was cancelled,
    // as partial files might still be desired or need proper closing.
    if (!writers.empty()) { // Check if writers were successfully created
         bool all_finalized = true;
         std::cout << "Info [C API]: Finalizing output writers..." << std::endl;
         for (size_t i = 0; i < writers.size(); ++i) {
             try { // Add try/catch around finalize
                 if (writers[i] && !writers[i]->finalize()) {
                     all_finalized = false;
                     std::cerr << "Error [C API]: Failed finalizing writer for " << (i < target_names.size() ? target_names[i] : "unknown") << std::endl;
                 }
             } catch (const std::exception& e) {
                 all_finalized = false;
                 std::cerr << "Error [C API]: Exception finalizing writer for " << (i < target_names.size() ? target_names[i] : "unknown") << ": " << e.what() << std::endl;
             } catch (...) {
                  all_finalized = false;
                  std::cerr << "Error [C API]: Unknown exception finalizing writer for " << (i < target_names.size() ? target_names[i] : "unknown") << std::endl;
             }
         }
         // Only override a SUCCESS code with a finalize error. Keep cancellation or earlier errors.
         if (!all_finalized && final_result == DEMUCS_RESULT_SUCCESS) {
             final_result = DEMUCS_RESULT_ERROR_WRITER_FINALIZE;
         }
    } else {
         std::cout << "Info [C API]: Skipping finalization as writers were not created." << std::endl;
    }

    // Final log based on the determined result code
    if (final_result == DEMUCS_RESULT_CANCELLED) { std::cerr << "Info [C API]: Processing was cancelled." << std::endl; }
    else if (final_result != DEMUCS_RESULT_SUCCESS) { std::cerr << "Info [C API]: Processing finished with error code: " << final_result << std::endl; }
    else { std::cout << "Info [C API]: Processing finished successfully." << std::endl; }

    return final_result; // Return the final determined result code

} // End of process_demucs_onnx_c
