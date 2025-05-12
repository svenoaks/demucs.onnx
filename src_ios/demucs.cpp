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
#include <limits>
#include <new>

// --- Project-Specific Includes ---
#include "demucs.hpp"       // Core C++ declarations (demucs_model, callbacks, constants, load_model declaration)
#include "demucs_result_codes.h" // Shared enum definition
#include "wav_writer.hpp"   // For StreamingWavWriter class
#include "wav_reader.hpp"   // For StreamingWavReader class
#include <onnxruntime/core/session/onnxruntime_cxx_api.h> // For ONNX Runtime
#include <unsupported/Eigen/CXX11/Tensor> // For Eigen::Tensor

// --- Include C Interface Header ---
#include "demucs_interface.h"




// --- Model loading (remains the same) ---
static demucsonnx::demucs_model load_model_internal(const std::string& htdemucs_model_path, Ort::SessionOptions& session_options) {
    demucsonnx::demucs_model model; std::ifstream file; std::vector<char> file_data;
    try {
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
    try {
        success = demucsonnx::load_model(file_data, model, session_options);
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

// --- Static writer helper (remains the same) ---
static void write_audio_chunk_internal(
                                       std::vector<std::unique_ptr<StreamingWavWriter>>& writers,
                                       std::vector<float>& interleaved_write_buffer,
                                       const Eigen::Tensor<float, 3>& chunk_data, // Expects (source, channel, sample)
                                       int num_valid_samples)
{
    if (num_valid_samples <= 0) {
        return;
    }

    const auto current_num_sources = chunk_data.dimension(0);
    const auto num_channels = chunk_data.dimension(1);
    const auto chunk_num_samples_dim = chunk_data.dimension(2);

    if (num_channels != 2) {
        std::cerr << "Error [write_audio_chunk_internal]: Expected 2 channels in chunk data, got " << num_channels << std::endl;
        return;
    }

    if (num_valid_samples > chunk_num_samples_dim) {
        std::cerr << "Warning [write_audio_chunk_internal]: num_valid_samples (" << num_valid_samples
        << ") exceeds chunk data dimension (" << chunk_num_samples_dim << "). Clamping." << std::endl;
        num_valid_samples = chunk_num_samples_dim;
        if (num_valid_samples <= 0) return;
    }

    const size_t required_buffer_size = static_cast<size_t>(num_valid_samples) * static_cast<size_t>(num_channels);
    try {
        if (interleaved_write_buffer.size() < required_buffer_size) {
            interleaved_write_buffer.resize(required_buffer_size);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error [write_audio_chunk_internal]: Failed to resize write buffer: " << e.what() << std::endl;
        return;
    }


    for (size_t i = 0; static_cast<Eigen::Index>(i) < current_num_sources; ++i) {
        if (i >= writers.size() || !writers[i]) {
            // Warning logged previously if needed
            continue;
        }

        bool bounds_error_occurred = false;
        for (int k = 0; k < num_valid_samples; ++k) {
            size_t base_idx = static_cast<size_t>(k) * static_cast<size_t>(num_channels);

            if (static_cast<Eigen::Index>(i) >= chunk_data.dimension(0) || 1 >= chunk_data.dimension(1)) {
                std::cerr << "Error [write_audio_chunk_internal]: Source/Channel index out of bounds during interleave (i=" << i << ", k=" << k << ")." << std::endl;
                for (size_t fill_idx = base_idx; fill_idx < required_buffer_size; ++fill_idx) {
                    if(fill_idx < interleaved_write_buffer.size()) interleaved_write_buffer[fill_idx] = 0.0f;
                }
                bounds_error_occurred = true;
                break;
            }

            interleaved_write_buffer[base_idx + 0] = chunk_data(i, 0, k);
            interleaved_write_buffer[base_idx + 1] = chunk_data(i, 1, k);
        }

        if (bounds_error_occurred) {
            continue;
        }

        if (!writers[i]->append_samples(interleaved_write_buffer.data(), required_buffer_size)) {
            std::cerr << "Error [write_audio_chunk_internal]: writer->append_samples failed for source index " << i << std::endl;
        }
    }
}

extern "C" DemucsResultCode process_demucs_onnx_c(
    const char* model_path_c,
    const char* input_wav_path_c,
    const char* output_dir_path_c,
    DemucsProgressCallback_C progress_callback_c,
    void* progress_callback_context,
    void* cancel_flag_context)
{
    /* --------------- Setup and validations --------------- */
    const auto* cancel = static_cast<const std::atomic<bool>*>(cancel_flag_context);
    if (!model_path_c || !input_wav_path_c || !output_dir_path_c || !cancel)
        return DEMUCS_RESULT_ERROR_INVALID_ARGS;
    if (cancel->load(std::memory_order_relaxed))
        return DEMUCS_RESULT_CANCELLED;

    StreamingWavReader reader;
    std::vector<std::unique_ptr<StreamingWavWriter>> writers;
    std::vector<std::string> stem_names;
    DemucsResultCode result = DEMUCS_RESULT_SUCCESS;
    drwav_uint64 total_input_frames = 0;
    long long total_written_orig = 0; // Tracks frames written to output files

    try {
        // Open input WAV and validate format
        if (!reader.open(input_wav_path_c))
            return DEMUCS_RESULT_ERROR_AUDIO_LOAD;
        total_input_frames = reader.get_total_frames();
        if (total_input_frames == 0) { reader.close(); return DEMUCS_RESULT_SUCCESS; }
        if (reader.get_channels() != 2 ||
            reader.get_sample_rate() != demucsonnx::SUPPORTED_SAMPLE_RATE) {
             std::cerr << "Error: Unsupported WAV format. Channels: " << reader.get_channels()
                       << " (expected 2), Sample Rate: " << reader.get_sample_rate()
                       << " (expected " << demucsonnx::SUPPORTED_SAMPLE_RATE << ")" << std::endl;
            reader.close();
            return DEMUCS_RESULT_ERROR_AUDIO_LOAD;
        }

        // Create output directory if needed
        std::filesystem::path out_dir(output_dir_path_c);
        std::error_code ec;
        // (Error handling for directory creation/validation remains the same)
        if (!std::filesystem::exists(out_dir)) {
            if (!std::filesystem::create_directories(out_dir, ec)) {
                 std::cerr << "Error: Cannot create output directory: " << out_dir << " (" << ec.message() << ")" << std::endl;
                reader.close();
                return DEMUCS_RESULT_ERROR_OUTPUT_DIR;
            }
        } else if (!std::filesystem::is_directory(out_dir)) {
             std::cerr << "Error: Output path exists but is not a directory: " << out_dir << std::endl;
            reader.close();
            return DEMUCS_RESULT_ERROR_OUTPUT_DIR;
        }

        // Load ONNX model
        Ort::SessionOptions so;
        so.DisableMemPattern();
        so.DisableCpuMemArena();
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency() / 2));
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        demucsonnx::demucs_model model = load_model_internal(model_path_c, so);
        if (model.nb_sources <= 0) {
            reader.close();
            return DEMUCS_RESULT_ERROR_MODEL_LOAD;
        }
        const int nb_stems = model.nb_sources;

        // Open output stem files
        // (Logic remains the same)
        const char* def_names[] = {"drums","bass","other","vocals","guitar","piano"};
        for (int s = 0; s < nb_stems; ++s) {
            std::string nm = (s < static_cast<int>(sizeof(def_names)/sizeof(def_names[0])) ? def_names[s] : "source_" + std::to_string(s));
            stem_names.push_back(nm);
            auto w = std::make_unique<StreamingWavWriter>();
            std::string out_path_str = (out_dir / (nm + ".wav")).string();
            if (!w->open(out_path_str, reader.get_sample_rate(), reader.get_channels())) {
                 std::cerr << "Error: Failed to open output file: " << out_path_str << std::endl;
                reader.close();
                for(auto& writer_to_close : writers) { if (writer_to_close) writer_to_close->finalize(); }
                return DEMUCS_RESULT_ERROR_WRITER_OPEN;
            }
            writers.push_back(std::move(w));
        }


        // Progress callback setup
        demucsonnx::ProgressCallback prog;
        // (Logic remains the same)
        if (progress_callback_c) {
            prog = [=](float p, const std::string& m) {
                if (!cancel->load(std::memory_order_relaxed)) {
                    progress_callback_c(progress_callback_context, p, m.c_str());
                }
            };
        } else {
            prog = [](float, const std::string&) {};
        }


        /* --------- Compute global mean and std for normalization --------- */
        // (Logic remains the same)
        double sum = 0.0, sumsq = 0.0;
        drwav_uint64 n = 0;
        const drwav_uint64 STAT_CHUNK = 44100 * 10;
        Eigen::MatrixXf statbuf;
        while (drwav_uint64 fr = reader.read_chunk(statbuf, STAT_CHUNK)) {
            if (fr > 0) {
                sum   += statbuf.leftCols(fr).sum();
                sumsq += statbuf.leftCols(fr).array().square().sum();
                n += fr * reader.get_channels();
            }
        }
        if (n == 0) {
             reader.close();
             for(auto& writer_to_close : writers) { if (writer_to_close) writer_to_close->finalize(); }
             return DEMUCS_RESULT_SUCCESS;
        }
        reader.seek_to_frame(0);
        float mean = static_cast<float>(sum / n);
        float stdv = std::sqrt(std::max(0.0, (sumsq / n) - static_cast<double>(mean * mean)));
        if (stdv < 1e-8f) stdv = 1.0f;

        /* ------------ Segment and overlap parameters ------------ */
        const int seg = int(demucsonnx::SEGMENT_LEN_SECS * reader.get_sample_rate());
        const int hop = int((1.0f - demucsonnx::OVERLAP) * seg); // Stride
        demucsonnx::demucs_segment_buffers segbuf(2, seg, nb_stems);
        demucsonnx::stft_buffers stftbuf(segbuf.padded_segment_samples);

        // REMOVED: Cross-fade weight window is no longer needed here
        // Eigen::VectorXf weight(seg);
        // ... weight calculation removed ...

        // Overlap-add buffer (circular)
        const int ring_len = seg + hop; // Use a buffer size that can hold a full segment + a hop
        Eigen::Tensor<float, 3> ola(nb_stems, 2, ring_len);
        ola.setZero();
        // REMOVED: wsum buffer is no longer needed
        // Eigen::VectorXf wsum(ring_len);
        // wsum.setZero();

        // Buffer for writing output and interleaving
        Eigen::Tensor<float, 3> processed_stride(nb_stems, 2, hop); // Holds one stride of output
        std::vector<float> interleaved_write_buffer;

        /* ---------------- Chunked processing (Simplified Overlap) ---------------- */
        const drwav_uint64 READ_CHUNK_FRAMES = 44100 * 30; // Process ~30 seconds chunks
        Eigen::MatrixXf input_chunk_buffer;                // Holds raw audio data read from file

        long long current_input_frame_global = 0; // Tracks global position *read* from the input file
        long long next_write_frame = 0;           // Tracks the next frame index to write to output files
        long long max_processed_frame = -1;       // Tracks highest frame index whose contribution has been *added* to OLA

        // Read and process the WAV in chunks
        while (true) {
            if (cancel->load(std::memory_order_relaxed)) {
                result = DEMUCS_RESULT_CANCELLED;
                break;
            }

            // --- Read next chunk ---
            drwav_uint64 frames_read_this_chunk = reader.read_chunk(input_chunk_buffer, READ_CHUNK_FRAMES);
            if (frames_read_this_chunk == 0 && current_input_frame_global >= (long long)total_input_frames) {
                break; // Truly end of file
            }
             if (frames_read_this_chunk == 0 && current_input_frame_global < (long long)total_input_frames) {
                 std::cerr << "Warning: WAV reader returned 0 frames before reaching expected end." << std::endl;
                 break; // Unexpected end or read error
             }

            long long process_start_frame = current_input_frame_global; // Monotonic start frame

            // --- Normalize and Process ---
            Eigen::MatrixXf normalized_chunk = (input_chunk_buffer.leftCols(frames_read_this_chunk).array() - mean) / stdv;

            // NOTE: Pass nullptr for the weight argument as it's no longer used
            result = demucsonnx::demucs_inference_process_chunk(
                model, normalized_chunk, process_start_frame,
                ola,
                segbuf, stftbuf, cancel
            );
            if (result != DEMUCS_RESULT_SUCCESS) {
                break; // Stop processing on error or cancel
            }

            // Update trackers
            max_processed_frame = std::max(max_processed_frame, process_start_frame + (long long)frames_read_this_chunk - 1);
            current_input_frame_global += frames_read_this_chunk; // Update after using old value

            // --- Flush fully processed segments from OLA to output ---
            // Read when the OLA buffer contains enough processed data for the next stride
            // The condition checks if the frame *at the end* of the hop has been processed.
            while (max_processed_frame >= next_write_frame + hop - 1)
            {
                 // Break if we've already written the whole file
                 if (next_write_frame >= (long long)total_input_frames) {
                      break;
                 }

                // Reconstruct one stride of audio (length = hop)
                for (int s = 0; s < nb_stems; ++s) {
                    for (int c = 0; c < 2; ++c) {
                        for (int k = 0; k < hop; ++k) {
                            long long frame_idx_global = next_write_frame + k;
                            int buffer_idx = frame_idx_global % ring_len;
                            if (buffer_idx < 0) buffer_idx += ring_len; // Ensure positive index

                            // Read directly from OLA buffer (no division by wsum)
                            float v = ola(s, c, buffer_idx);

                            // Clamp k to ensure it's within processed_stride bounds
                            if (k < processed_stride.dimension(2)) {
                               processed_stride(s, c, k) = v * stdv + mean; // Denormalize
                            } else {
                                // Should not happen if hop <= processed_stride.dimension(2)
                                std::cerr << "Warning: Index k (" << k << ") out of bounds for processed_stride." << std::endl;
                            }

                            // Clear the OLA buffer slots as they are read
                            ola(s, c, buffer_idx) = 0.0f;
                        }
                    }
                }
                // REMOVED: Clearing wsum buffer is no longer needed

                // Determine valid range to write (clamp to total_input_frames)
                long long remaining_output_frames = (long long) total_input_frames - next_write_frame;
                int valid_samples_in_stride = std::min<long long>(hop, std::max<long long>(0, remaining_output_frames));

                if (valid_samples_in_stride > 0) {
                     // Ensure processed_stride has enough columns before slicing
                     if (valid_samples_in_stride <= processed_stride.dimension(2)) {
                          Eigen::Tensor<float, 3> slice =
                              processed_stride.slice(Eigen::array<Eigen::Index, 3>{0, 0, 0},
                                                      Eigen::array<Eigen::Index, 3>{nb_stems, 2, valid_samples_in_stride});
                          write_audio_chunk_internal(writers, interleaved_write_buffer, slice, valid_samples_in_stride);
                          total_written_orig += valid_samples_in_stride;
                     } else {
                          std::cerr << "Error: Logic error - valid_samples_in_stride (" << valid_samples_in_stride
                                    << ") exceeds processed_stride buffer size (" << processed_stride.dimension(2) << ")" << std::endl;
                     }
                }

                // Advance the write pointer
                next_write_frame += hop;

                // Update progress
                if (prog && total_input_frames > 0) {
                    prog(float(total_written_orig) / float(total_input_frames), "Processing");
                }
                 // Break inner loop if done
                 if (total_written_orig >= (long long)total_input_frames) {
                      break;
                 }
            } // end while writing strides

            // Break outer loop if done
            if (total_written_orig >= (long long)total_input_frames) {
                break;
            }

        } // end while reading chunks

        // Check for cancellation/errors after main loop
        if (result != DEMUCS_RESULT_SUCCESS) {
             reader.close();
             for (auto &w : writers) { if (w) w->finalize(); }
             return result;
        }

        /* ---------------- Final Flush ---------------- */
        // This loop ensures any remaining frames up to total_input_frames are written.
        while (total_written_orig < (long long) total_input_frames)
        {
             // Check if data is ready in OLA
             if (max_processed_frame < next_write_frame + hop - 1) {
                  // Data might not be fully ready if the input ended abruptly or processing stalled.
                  // Break and potentially leave the output slightly truncated.
                  if (next_write_frame < (long long)total_input_frames) { // Only warn if we haven't reached the end yet
                     std::cerr << "Warning: Final flush loop stalled waiting for data. Max processed: " << max_processed_frame
                               << ", Next write start: " << next_write_frame << ". Output might be short." << std::endl;
                  }
                  break;
             }

             // Reconstruct one stride
             for (int s = 0; s < nb_stems; ++s) {
                 for (int c = 0; c < 2; ++c) {
                     for (int k = 0; k < hop; ++k) {
                         long long frame_idx_global = next_write_frame + k;
                         int buffer_idx = frame_idx_global % ring_len;
                         if (buffer_idx < 0) buffer_idx += ring_len;

                         // Read directly, no wsum division
                         float v = ola(s, c, buffer_idx);

                         if (k < processed_stride.dimension(2)) {
                            processed_stride(s, c, k) = v * stdv + mean; // Denormalize
                         }
                         ola(s, c, buffer_idx) = 0.0f; // Clear OLA
                     }
                 }
             }
             // REMOVED: Clearing wsum buffer

             // Determine valid range (clamp to total_input_frames)
             long long remaining_output_frames = (long long) total_input_frames - next_write_frame;
             int valid_samples_in_stride = std::min<long long>(hop, std::max<long long>(0, remaining_output_frames));

             if (valid_samples_in_stride > 0) {
                  if (valid_samples_in_stride <= processed_stride.dimension(2)) {
                      Eigen::Tensor<float, 3> slice =
                          processed_stride.slice(Eigen::array<Eigen::Index, 3>{0, 0, 0},
                                                  Eigen::array<Eigen::Index, 3>{nb_stems, 2, valid_samples_in_stride});
                      write_audio_chunk_internal(writers, interleaved_write_buffer, slice, valid_samples_in_stride);
                      total_written_orig += valid_samples_in_stride;
                  } else {
                       std::cerr << "Error: Logic error in final flush - valid_samples_in_stride (" << valid_samples_in_stride
                                 << ") exceeds buffer size (" << processed_stride.dimension(2) << ")" << std::endl;
                  }
             }

             // Advance write pointer
             next_write_frame += hop;

             // Update progress
             if (prog && total_input_frames > 0) {
                 prog(float(total_written_orig) / float(total_input_frames), "Finalizing");
             }
        } // end final flush loop

        // Final progress update
        if (prog && total_input_frames > 0) {
             prog(1.0f, "Done");
        }

    } catch (const std::bad_alloc& e) { // Catch memory allocation errors specifically
        std::cerr << "[process_demucs_onnx_c] Memory allocation Exception: " << e.what() << std::endl;
        result = DEMUCS_RESULT_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        std::cerr << "[process_demucs_onnx_c] Exception: " << e.what() << std::endl;
        result = DEMUCS_RESULT_ERROR_CPP_EXCEPTION;
    } catch (...) {
        std::cerr << "[process_demucs_onnx_c] Unknown exception occurred." << std::endl;
        result = DEMUCS_RESULT_ERROR_CPP_EXCEPTION;
    }

    // --- Cleanup ---
    reader.close();
    for (auto &w : writers) {
        if (w) w->finalize();
    }

     // Final check on written frames
     if (total_written_orig < (long long)total_input_frames && result == DEMUCS_RESULT_SUCCESS) {
         std::cerr << "Warning: Final output length (" << total_written_orig
                   << ") is shorter than input length (" << total_input_frames << ")." << std::endl;
     } else if (total_written_orig > (long long)total_input_frames) {
          std::cerr << "Warning: Final output length (" << total_written_orig
                    << ") is longer than input length (" << total_input_frames << ")." << std::endl;
     }

    return result;
}
