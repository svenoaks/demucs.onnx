#include "demucs.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>

// At global/class scope
static std::random_device rd;  // Get entropy for seed
static std::mt19937 gen(rd()); // Mersenne Twister generator

namespace demucsonnx {

static void reflect_padding(
                            Eigen::MatrixXf &padded_mix,
                            int left_padding,
                            int right_padding,
                            int N)
{
    // Reflect from the first 'left_padding' samples of the original data
    for (int i = 0; i < left_padding; ++i)
    {
        padded_mix.block(0, left_padding - 1 - i, 2, 1) =
        padded_mix.block(0, left_padding + i, 2, 1);
    }
    
    // Reflect from the last 'right_padding' samples of the original data
    for (int i = 0; i < right_padding; ++i)
    {
        int last_elem = N + left_padding - 1 ;
        padded_mix.block(0, last_elem + i + 1, 2, 1) =
        padded_mix.block(0, last_elem - i, 2, 1);
    }
}

// In-Memory Incremental Inference function
DemucsResultCode demucs_inference_incremental(
                                              struct demucsonnx::demucs_model &model,
                                              const Eigen::MatrixXf &audio,
                                              demucsonnx::ProgressCallback cb,
                                              const demucsonnx::WriteChunkCallback& write_callback,
                                              const std::atomic<bool>* cancel_flag
                                              )
{
    // --- Initial Cancellation Check ---
    if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
        return DEMUCS_RESULT_CANCELLED;
    }
    
    const int length = audio.cols();
    const int nb_out_sources = model.nb_sources;
    const int num_channels = 2; // Assuming stereo pipeline
    
    if (length == 0) {
        if(cb) cb(1.0f, "Input audio is empty.");
        return DEMUCS_RESULT_SUCCESS; // Nothing to process
    }
    
    // --- Normalization, Padding, Shifting (applied to full audio) ---
    const int max_shift = static_cast<int>(demucsonnx::MAX_SHIFT_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);
    Eigen::VectorXf ref_mean_0 = audio.colwise().mean();
    const float ref_mean = ref_mean_0.mean();
    float variance = (audio.array() - ref_mean).square().sum() / (audio.size() > 1 ? (audio.size() - 1) : 1);
    variance = std::max(variance, 0.0f);
    const float ref_std = (variance > 1e-8f) ? std::sqrt(variance) : 1.0f;
    const int padded_length = length + 2 * max_shift;
    Eigen::MatrixXf padded_mix(num_channels, padded_length);
    padded_mix.setZero();
    padded_mix.block(0, max_shift, num_channels, length) = (audio.array() - ref_mean) / ref_std;
    std::uniform_int_distribution<> dist(0, max_shift > 0 ? max_shift - 1 : 0);
    const int shift_offset = (max_shift > 0) ? dist(gen) : 0;
    const int shifted_length = length + max_shift - shift_offset;
    Eigen::Ref<const Eigen::MatrixXf> shifted_audio = padded_mix.block(0, shift_offset, num_channels, shifted_length);
    
    // --- Segment and Overlap Parameters ---
    const int segment_samples = static_cast<int>(demucsonnx::SEGMENT_LEN_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);
    const int stride_samples = static_cast<int>((1.0f - demucsonnx::OVERLAP) * segment_samples);
    if (segment_samples <= 0 || stride_samples <= 0) {
        std::cerr << "Error [inference]: Invalid segment/stride size calculated (segment=" << segment_samples << ", stride=" << stride_samples << ")." << std::endl;
        return DEMUCS_RESULT_ERROR_UNKNOWN; // Or a specific config error
    }
    
    // --- Buffers ---
    demucsonnx::demucs_segment_buffers buffers(num_channels, segment_samples, nb_out_sources);
    demucsonnx::stft_buffers stft_buf(buffers.padded_segment_samples);
    Eigen::VectorXf weight(segment_samples);
    int half_segment = segment_samples / 2;
    weight.head(half_segment) = Eigen::VectorXf::LinSpaced(half_segment, 1, half_segment);
    weight.tail(segment_samples - half_segment) = weight.head(half_segment).reverse();
    float max_weight = weight.maxCoeff();
    if (max_weight > 1e-8f) { weight /= max_weight; } else { weight.setOnes(); }
    weight = weight.array().pow(demucsonnx::TRANSITION_POWER);
    
    // --- Incremental Output Buffering ---
    const int output_buffer_len = 2 * segment_samples;
    Eigen::Tensor<float, 3> output_buffer(nb_out_sources, num_channels, output_buffer_len);
    Eigen::VectorXf sum_weight_buffer(output_buffer_len);
    output_buffer.setZero(); sum_weight_buffer.setZero();
    long long buffer_start_sample_idx = 0;
    long long samples_ready_to_write = 0;
    Eigen::Tensor<float, 3> processed_chunk(nb_out_sources, num_channels, stride_samples);
    
    // --- Main Processing Loop ---
    const int total_segments = static_cast<int>(std::ceil(static_cast<float>(shifted_length) / stride_samples));
    float progress = 0.0f;
    const float progress_increment = (total_segments > 0) ? (1.0f / static_cast<float>(total_segments)) : 0.0f;
    
    for (long long segment_offset = 0; segment_offset < shifted_length; segment_offset += stride_samples)
    {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return DEMUCS_RESULT_CANCELLED;
        
        // --- Prepare Input Segment ---
        const long long current_segment_length = std::min((long long)segment_samples, shifted_length - segment_offset);
        if (current_segment_length <= 0) continue;
        Eigen::Ref<const Eigen::MatrixXf> segment_chunk = shifted_audio.block(0, segment_offset, num_channels, current_segment_length);
        const int symmetric_padding = buffers.padded_segment_samples - current_segment_length;
        const int symmetric_padding_start = symmetric_padding / 2;
        buffers.padded_mix.setZero();
        buffers.padded_mix.block(0, buffers.pad + symmetric_padding_start, num_channels, current_segment_length) = segment_chunk;
        reflect_padding(buffers.padded_mix, buffers.pad, buffers.pad_end, current_segment_length); // Call external/linked function
        
        // --- Run Model Inference (with corrected catch block) ---
        try {
            // Call the function containing the actual ONNX Runtime Run() call
            demucsonnx::model_inference(model, buffers, stft_buf); // Call external/linked function
        } catch (const std::exception& e) { // Catch standard exceptions
            std::cerr << "Error [inference]: std::exception during model_inference: " << e.what() << std::endl;
            return DEMUCS_RESULT_ERROR_INFERENCE;
        } catch (...) { // Catch any other type of exception
            std::cerr << "Error [inference]: Unknown exception during model_inference." << std::endl;
            return DEMUCS_RESULT_ERROR_INFERENCE; // Return specific code for ANY inference error
        }
        
        // --- Accumulate Weighted Output ---
        long long current_segment_abs_idx = segment_offset;
        long long buffer_relative_idx = current_segment_abs_idx - buffer_start_sample_idx;
        buffer_relative_idx = std::max(0LL, buffer_relative_idx);
        for (int i = 0; i < nb_out_sources; ++i) {
            for (int j = 0; j < num_channels; ++j) {
                for (int k = 0; k < current_segment_length; ++k) {
                    int model_output_k_idx = k + symmetric_padding_start;
                    model_output_k_idx = std::max(0, std::min(model_output_k_idx, (int)buffers.targets_out.dimension(2) - 1));
                    long long buffer_k_idx = buffer_relative_idx + k;
                    if (buffer_k_idx >= 0 && buffer_k_idx < output_buffer_len) {
                        int weight_idx = std::max(0, std::min(k, (int)weight.size() - 1));
                        if (weight_idx >= 0) {
                            output_buffer(i, j, buffer_k_idx) += weight(weight_idx) * buffers.targets_out(i, j, model_output_k_idx);
                        }
                    }
                }
            }
        }
        for (int k = 0; k < current_segment_length; ++k) {
            long long buffer_k_idx = buffer_relative_idx + k;
            if (buffer_k_idx >= 0 && buffer_k_idx < output_buffer_len) {
                int weight_idx = std::max(0, std::min(k, (int)weight.size() - 1));
                if (weight_idx >= 0) { sum_weight_buffer(buffer_k_idx) += weight(weight_idx); }
            }
        }
        
        // --- Check if a Stride Block is Ready ---
        samples_ready_to_write += stride_samples;
        if (samples_ready_to_write >= stride_samples) {
            const int num_samples_to_process_from_buffer = stride_samples;
            long long ready_block_start_in_shifted = buffer_start_sample_idx;
            long long ready_block_end_in_shifted = ready_block_start_in_shifted + num_samples_to_process_from_buffer;
            long long original_first_idx = ready_block_start_in_shifted - (max_shift - shift_offset);
            long long original_last_idx = ready_block_end_in_shifted - (max_shift - shift_offset);
            int actual_samples_this_block = 0;
            if (original_last_idx >= 0 && original_first_idx < length) {
                long long write_start_orig = std::max(0LL, original_first_idx);
                long long write_end_orig = std::min((long long)length - 1, original_last_idx);
                actual_samples_this_block = static_cast<int>(write_end_orig - write_start_orig + 1);
            }
            
            if (actual_samples_this_block > 0) {
                long long write_start_orig = std::max(0LL, original_first_idx);
                int buffer_read_start_idx = static_cast<int>(write_start_orig + (max_shift - shift_offset) - buffer_start_sample_idx);
                buffer_read_start_idx = std::max(0, buffer_read_start_idx);
                if (buffer_read_start_idx + actual_samples_this_block > num_samples_to_process_from_buffer) {
                    actual_samples_this_block = std::max(0, num_samples_to_process_from_buffer - buffer_read_start_idx);
                }
                
                if (actual_samples_this_block > 0) {
                    if (processed_chunk.dimension(2) != actual_samples_this_block) {
                        processed_chunk.resize(nb_out_sources, num_channels, actual_samples_this_block);
                    }
                    for (int i = 0; i < nb_out_sources; ++i) {
                        for (int j = 0; j < num_channels; ++j) {
                            for (int k = 0; k < actual_samples_this_block; ++k) {
                                int buffer_idx = buffer_read_start_idx + k;
                                float val = 0.0f;
                                if (buffer_idx >= 0 && buffer_idx < output_buffer_len) {
                                    if (buffer_idx < sum_weight_buffer.size() && sum_weight_buffer(buffer_idx) > 1e-8f) {
                                        val = output_buffer(i, j, buffer_idx) / sum_weight_buffer(buffer_idx);
                                    } else { val = output_buffer(i, j, buffer_idx); }
                                }
                                val = val * ref_std + ref_mean;
                                processed_chunk(i, j, k) = val;
                            }
                        }
                    }
                    try { write_callback(processed_chunk, actual_samples_this_block); }
                    catch (const std::exception& e) { std::cerr << "Error [inference]: Exception during write callback: " << e.what() << std::endl; return DEMUCS_RESULT_ERROR_WRITER_FINALIZE; }
                    catch (...) { std::cerr << "Error [inference]: Unknown exception during write callback." << std::endl; return DEMUCS_RESULT_ERROR_WRITER_FINALIZE; }
                }
            }
            
            // Shift the Rolling Buffers
            if (num_samples_to_process_from_buffer < output_buffer_len) {
                Eigen::TensorMap<Eigen::Tensor<float, 3>> buf_map(output_buffer.data(), nb_out_sources, num_channels, output_buffer_len);
                Eigen::TensorMap<Eigen::Tensor<float, 1>> sw_map(sum_weight_buffer.data(), output_buffer_len);
                int remaining = output_buffer_len - num_samples_to_process_from_buffer;
                Eigen::array<Eigen::Index, 3> src_o={0,0,num_samples_to_process_from_buffer}, dst_o={0,0,0}, ext={nb_out_sources,num_channels,remaining};
                Eigen::array<Eigen::Index, 1> sw_src_o={num_samples_to_process_from_buffer}, sw_dst_o={0}, sw_ext={remaining};
                buf_map.slice(dst_o, ext) = buf_map.slice(src_o, ext);
                sw_map.slice(sw_dst_o, sw_ext) = sw_map.slice(sw_src_o, sw_ext);
                Eigen::array<Eigen::Index, 3> zero_o={0,0,remaining}, zero_ext={nb_out_sources,num_channels,num_samples_to_process_from_buffer};
                Eigen::array<Eigen::Index, 1> sw_zero_o={remaining}, sw_zero_ext={num_samples_to_process_from_buffer};
                buf_map.slice(zero_o, zero_ext).setZero();
                sw_map.slice(sw_zero_o, sw_zero_ext).setZero();
            } else { output_buffer.setZero(); sum_weight_buffer.setZero(); }
            buffer_start_sample_idx += num_samples_to_process_from_buffer;
            samples_ready_to_write -= num_samples_to_process_from_buffer;
        } // end if samples_ready_to_write
        
        // --- Update Progress ---
        progress += progress_increment;
        if(cb) cb(std::min(1.0f, progress), "Processing segment");
        
    } // End segment loop
    
    if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) return DEMUCS_RESULT_CANCELLED;
    
    // --- Handle Remaining Samples ---
    long long remaining_in_buffer = (shifted_length - buffer_start_sample_idx);
    remaining_in_buffer = std::min(remaining_in_buffer, (long long)output_buffer_len);
    if (remaining_in_buffer > 0) {
        long long start_shifted = buffer_start_sample_idx;
        long long end_shifted = start_shifted + remaining_in_buffer;
        long long start_orig = start_shifted - (max_shift - shift_offset);
        long long end_orig = end_shifted - (max_shift - shift_offset);
        int actual_to_write = 0;
        if (end_orig >= 0 && start_orig < length) {
            long long write_start = std::max(0LL, start_orig);
            long long write_end = std::min((long long)length - 1, end_orig);
            actual_to_write = static_cast<int>(write_end - write_start + 1);
        }
        
        if (actual_to_write > 0) {
            long long write_start = std::max(0LL, start_orig);
            int buffer_read_start = static_cast<int>(write_start + (max_shift - shift_offset) - buffer_start_sample_idx);
            buffer_read_start = std::max(0, buffer_read_start);
            if (buffer_read_start + actual_to_write > remaining_in_buffer) {
                actual_to_write = std::max(0LL, remaining_in_buffer - buffer_read_start);
            }
            
            if (actual_to_write > 0) {
                if (processed_chunk.dimension(2) != actual_to_write) { processed_chunk.resize(nb_out_sources, num_channels, actual_to_write); }
                for (int i = 0; i < nb_out_sources; ++i) {
                    for (int j = 0; j < num_channels; ++j) {
                        for (int k = 0; k < actual_to_write; ++k) {
                            int buffer_idx = buffer_read_start + k;
                            float val = 0.0f;
                            if (buffer_idx >= 0 && buffer_idx < output_buffer_len) {
                                if (buffer_idx < sum_weight_buffer.size() && sum_weight_buffer(buffer_idx) > 1e-8f) {
                                    val = output_buffer(i, j, buffer_idx) / sum_weight_buffer(buffer_idx);
                                } else { val = output_buffer(i, j, buffer_idx); }
                            }
                            val = val * ref_std + ref_mean;
                            processed_chunk(i, j, k) = val;
                        }
                    }
                }
                try { write_callback(processed_chunk, actual_to_write); }
                catch (const std::exception& e) { std::cerr << "Error [inference]: Exception during final write callback: " << e.what() << std::endl; return DEMUCS_RESULT_ERROR_WRITER_FINALIZE; }
                catch (...) { std::cerr << "Error [inference]: Unknown exception during final write callback." << std::endl; return DEMUCS_RESULT_ERROR_WRITER_FINALIZE; }
            }
        }
    }
    
    if(cb) cb(1.0f, "Inference finished");
    return DEMUCS_RESULT_SUCCESS;
} // End demucs_inference_incremental

}
