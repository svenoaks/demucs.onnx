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

Eigen::Tensor3dXf demucsonnx::demucs_inference(
    struct demucsonnx::demucs_model &model,
    const Eigen::MatrixXf &audio,
    demucsonnx::ProgressCallback cb)
{
    int length = audio.cols();
    int max_shift = static_cast<int>(demucsonnx::MAX_SHIFT_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);

    // Calculate the overall mean and standard deviation
    Eigen::VectorXf ref_mean_0 = audio.colwise().mean();
    float ref_mean = ref_mean_0.mean();
    float ref_std = std::sqrt((ref_mean_0.array() - ref_mean).square().sum() /
                              (ref_mean_0.size() - 1));

    // Create padded_mix with symmetric zero padding
    int padded_length = length + 2 * max_shift;
    Eigen::MatrixXf padded_mix(2, padded_length);
    padded_mix.setZero();

    // Normalize and copy audio into padded_mix starting at column max_shift
    padded_mix.block(0, max_shift, 2, length) = (audio.array() - ref_mean) / ref_std;

    // Generate random shift offset for time invariance
    std::uniform_int_distribution<> dist(0, max_shift - 1);
    int shift_offset = dist(gen);
    std::cout << "shift offset is: " << shift_offset << std::endl;

    // Create a block view for shifted_audio to avoid extra allocation
    int shifted_length = length + max_shift - shift_offset;
    Eigen::Ref<const Eigen::MatrixXf> shifted_audio = padded_mix.block(0, shift_offset, 2, shifted_length);

    // --- Begin merged split_inference and segment_inference logic ---

    // Calculate segment size in samples
    int segment_samples = static_cast<int>(demucsonnx::SEGMENT_LEN_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);
    int nb_out_sources = model.nb_sources;

    // Create reusable buffers with padded sizes
    demucsonnx::demucs_segment_buffers buffers(2, segment_samples, nb_out_sources);
    demucsonnx::stft_buffers stft_buf(buffers.padded_segment_samples);

    // Calculate stride for overlapping segments
    int stride_samples = static_cast<int>((1.0f - demucsonnx::OVERLAP) * segment_samples);

    // Create an output tensor initialized to zero
    Eigen::Tensor3dXf out(nb_out_sources, 2, shifted_length);
    out.setZero();

    // Create weight vector for overlapping segments
    Eigen::VectorXf weight(segment_samples);
    int half_segment = segment_samples / 2;
    weight.head(half_segment) = Eigen::VectorXf::LinSpaced(half_segment, 1, half_segment);
    weight.tail(half_segment) = weight.head(half_segment).reverse();
    weight /= weight.maxCoeff();
    weight = weight.array().pow(demucsonnx::TRANSITION_POWER);

    // Initialize sum_weight as Eigen::VectorXf
    Eigen::VectorXf sum_weight(shifted_length);
    sum_weight.setZero();

    // Calculate total number of chunks
    int total_chunks = static_cast<int>(std::ceil(static_cast<float>(shifted_length) / stride_samples));
    float increment_per_chunk = 1.0f / static_cast<float>(total_chunks);
    float inference_progress = 0.0f;

    // Preallocate a tensor for chunk_out to avoid repeated allocations
    Eigen::Tensor3dXf chunk_out(nb_out_sources, 2, segment_samples);
    chunk_out.setZero();

    // Process each segment
    for (int segment_offset = 0; segment_offset < shifted_length; segment_offset += stride_samples)
    {
        int chunk_length = std::min(segment_samples, shifted_length - segment_offset);

        // Create a block view for the current chunk
        Eigen::Ref<const Eigen::MatrixXf> chunk = shifted_audio.block(0, segment_offset, 2, chunk_length);

        // first, symmetric padding with zeros to fit the smaller chunk
        // into the bigger segment_samples
        int symmetric_padding = buffers.padded_segment_samples - chunk_length;

        buffers.padded_mix.setZero();
        // copy chunk into padded_mix at position buffers.pad + symmetric_padding/2
        int symmetric_padding_start = symmetric_padding / 2;
        buffers.padded_mix.block(0, buffers.pad + symmetric_padding_start, 2, chunk_length) = chunk;

        // then, reflect padding on the left and right for
        // the stft boundary effects
        reflect_padding(buffers.padded_mix, buffers.pad, buffers.pad_end, segment_samples);

        // Run model inference
        demucsonnx::model_inference(model, buffers, stft_buf);

        // Update progress
        cb(inference_progress + increment_per_chunk, "Segment inference complete");

        // Copy from buffers.targets_out into chunk_out with center trimming
        // Preallocate chunk_out and set to zero
        chunk_out.setZero();

        for (int i = 0; i < nb_out_sources; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < chunk_length; ++k)
                {
                    auto kidx = k + symmetric_padding_start;
                    kidx = std::min(kidx, int(buffers.targets_out.dimension(2)-1));
                    // Undoing center_trim by offsetting with left_padding
                    chunk_out(i, j, k) = buffers.targets_out(i, j, kidx);
                }
            }
        }

        // Accumulate the weighted chunk output using Eigen::Map for vectorization
        for (int i = 0; i < nb_out_sources; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                // Map the (i, j, chunk_length) slice of 'out' to an Eigen::ArrayXf
                Eigen::Map<Eigen::ArrayXf> out_slice(&out(i, j, segment_offset), chunk_length);
                // Map the (i, j, chunk_length) slice of 'chunk_out' to an Eigen::ArrayXf
                Eigen::Map<const Eigen::ArrayXf> chunk_out_slice(&chunk_out(i, j, 0), chunk_length);
                // Accumulate the weighted chunk output
                out_slice += weight.head(chunk_length).array() * chunk_out_slice;
            }
        }

        // Accumulate the weights using Eigen's segment operation
        sum_weight.segment(segment_offset, chunk_length) += weight.head(chunk_length);

        inference_progress += increment_per_chunk;
    }

    // Normalize the output by sum_weight using vectorized operations
    for (int i = 0; i < nb_out_sources; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            // Map the (i, j, shifted_length) slice of 'out' to an Eigen::ArrayXf
            Eigen::Map<Eigen::ArrayXf> out_slice(&out(i, j, 0), shifted_length);
            // Perform element-wise division
            out_slice /= sum_weight.array();
        }
    }

    // Inverse the normalization directly on 'out' using vectorized operations
    for (int i = 0; i < nb_out_sources; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            // Map the (i, j, shifted_length) slice of 'out' to an Eigen::ArrayXf
            Eigen::Map<Eigen::ArrayXf> out_slice(&out(i, j, 0), shifted_length);
            // Perform inverse normalization
            out_slice = out_slice * ref_std + ref_mean;
        }
    }

    // Create a view (slice) of 'out' without allocating new memory
    int trim_start = max_shift - shift_offset;
    Eigen::array<Eigen::Index, 3> offset_array = {0, 0, trim_start};
    Eigen::array<Eigen::Index, 3> extent_array = {nb_out_sources, 2, length};
    auto result = out.slice(offset_array, extent_array);

    // Materialize the slice into a new tensor to return
    Eigen::Tensor3dXf trimmed_waveform_outputs = result.eval();

    return trimmed_waveform_outputs;
}

void demucsonnx::demucs_inference_incremental(
    struct demucsonnx::demucs_model &model,
    const Eigen::MatrixXf &audio, // Original audio
    demucsonnx::ProgressCallback cb,
    const demucsonnx::WriteChunkCallback& write_callback // Accept the callback
    )
{
    const int length = audio.cols();
    const int nb_out_sources = model.nb_sources;
    const int num_channels = 2; // Assuming stereo

    // --- Normalization, Padding, Shifting ---
    const int max_shift = static_cast<int>(demucsonnx::MAX_SHIFT_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);
    Eigen::VectorXf ref_mean_0 = audio.colwise().mean();
    const float ref_mean = ref_mean_0.mean();
    const float ref_std = std::sqrt((ref_mean_0.array() - ref_mean).square().sum() / (ref_mean_0.size() - 1));
    const int padded_length = length + 2 * max_shift;
    Eigen::MatrixXf padded_mix(num_channels, padded_length);
    padded_mix.setZero();
    padded_mix.block(0, max_shift, num_channels, length) = (audio.array() - ref_mean) / ref_std;
    std::uniform_int_distribution<> dist(0, max_shift - 1);
    const int shift_offset = dist(gen);
    std::cout << "Shift offset: " << shift_offset << std::endl;
    const int shifted_length = length + max_shift - shift_offset;
    Eigen::Ref<const Eigen::MatrixXf> shifted_audio = padded_mix.block(0, shift_offset, num_channels, shifted_length);

    // --- Segment and Overlap Parameters ---
    const int segment_samples = static_cast<int>(demucsonnx::SEGMENT_LEN_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);
    const int stride_samples = static_cast<int>((1.0f - demucsonnx::OVERLAP) * segment_samples);

    // --- Buffers ---
    demucsonnx::demucs_segment_buffers buffers(num_channels, segment_samples, nb_out_sources);
    demucsonnx::stft_buffers stft_buf(buffers.padded_segment_samples);
    Eigen::VectorXf weight(segment_samples);
    int half_segment = segment_samples / 2;
    weight.head(half_segment) = Eigen::VectorXf::LinSpaced(half_segment, 1, half_segment);
    weight.tail(half_segment) = weight.head(half_segment).reverse();
    weight /= weight.maxCoeff();
    weight = weight.array().pow(demucsonnx::TRANSITION_POWER);

    // --- Incremental Output Buffering ---
    const int output_buffer_len = 2 * segment_samples;
    Eigen::Tensor<float, 3> output_buffer(nb_out_sources, num_channels, output_buffer_len);
    Eigen::VectorXf sum_weight_buffer(output_buffer_len);
    output_buffer.setZero();
    sum_weight_buffer.setZero();
    long long buffer_start_sample_idx = 0;
    long long samples_ready_to_write = 0;

    // --- Temporary tensor for holding the chunk to be passed to the callback ---
    Eigen::Tensor<float, 3> processed_chunk(nb_out_sources, num_channels, stride_samples);


    // --- Main Processing Loop ---
    const int total_chunks = static_cast<int>(std::ceil(static_cast<float>(shifted_length) / stride_samples));
    float progress = 0.0f;
    const float progress_increment = 1.0f / total_chunks;

    for (int segment_offset = 0; segment_offset < shifted_length; segment_offset += stride_samples)
    {
        // --- Prepare Input Chunk ---
        const int chunk_length = std::min(segment_samples, shifted_length - segment_offset);
        Eigen::Ref<const Eigen::MatrixXf> chunk = shifted_audio.block(0, segment_offset, num_channels, chunk_length);
        const int symmetric_padding = buffers.padded_segment_samples - chunk_length;
        const int symmetric_padding_start = symmetric_padding / 2;
        buffers.padded_mix.setZero();
        buffers.padded_mix.block(0, buffers.pad + symmetric_padding_start, num_channels, chunk_length) = chunk;
        reflect_padding(buffers.padded_mix, buffers.pad, buffers.pad_end, segment_samples);

        // --- Run Model Inference ---
        demucsonnx::model_inference(model, buffers, stft_buf);

        // --- Accumulate Weighted Output ---
        long long current_segment_abs_idx = segment_offset;
        long long buffer_relative_idx = current_segment_abs_idx - buffer_start_sample_idx;
        buffer_relative_idx = std::max(0LL, buffer_relative_idx);

        for (int i = 0; i < nb_out_sources; ++i) {
            for (int j = 0; j < num_channels; ++j) {
                for (int k = 0; k < chunk_length; ++k) {
                    int model_output_k_idx = k + symmetric_padding_start;
                    model_output_k_idx = std::min(model_output_k_idx, (int)buffers.targets_out.dimension(2) - 1);
                    long long buffer_k_idx = buffer_relative_idx + k;
                    if (buffer_k_idx >= 0 && buffer_k_idx < output_buffer_len) {
                        output_buffer(i, j, buffer_k_idx) += weight(k) * buffers.targets_out(i, j, model_output_k_idx);
                    }
                }
            }
        }
        for (int k = 0; k < chunk_length; ++k) {
             long long buffer_k_idx = buffer_relative_idx + k;
             if (buffer_k_idx >= 0 && buffer_k_idx < output_buffer_len) {
                 sum_weight_buffer(buffer_k_idx) += weight(k);
             }
        }

        // --- Check if a Stride Block is Ready ---
        samples_ready_to_write += stride_samples;

        if (samples_ready_to_write >= stride_samples) {
             const int num_samples_to_process = stride_samples;
             long long first_sample_abs_idx_to_write = buffer_start_sample_idx;
             long long last_sample_abs_idx_to_write = buffer_start_sample_idx + num_samples_to_process - 1;
             long long original_first_idx = first_sample_abs_idx_to_write - (max_shift - shift_offset);
             long long original_last_idx = last_sample_abs_idx_to_write - (max_shift - shift_offset);
             int actual_samples_this_block = 0; // Declare here for this block

             if (original_last_idx >= 0 && original_first_idx < length) {
                 long long write_start_orig = std::max(0LL, original_first_idx);
                 long long write_end_orig = std::min((long long)length - 1, original_last_idx);
                 actual_samples_this_block = static_cast<int>(write_end_orig - write_start_orig + 1);

                 if (actual_samples_this_block > 0) {
                      int buffer_write_start_idx = static_cast<int>(write_start_orig + (max_shift - shift_offset) - buffer_start_sample_idx);
                      // Ensure start index is valid within the buffer
                      buffer_write_start_idx = std::max(0, buffer_write_start_idx);


                      // --- Prepare data chunk for callback ---
                      if (processed_chunk.dimension(2) != actual_samples_this_block) {
                          processed_chunk.resize(nb_out_sources, num_channels, actual_samples_this_block);
                      }

                      for (int i = 0; i < nb_out_sources; ++i) {
                          for (int j = 0; j < num_channels; ++j) {
                               for (int k = 0; k < actual_samples_this_block; ++k) {
                                   int buffer_idx = buffer_write_start_idx + k;
                                   float val = 0.0f;
                                   if (buffer_idx >= 0 && buffer_idx < sum_weight_buffer.size() && sum_weight_buffer(buffer_idx) > 1e-8f) {
                                       val = output_buffer(i, j, buffer_idx) / sum_weight_buffer(buffer_idx);
                                   } else if (buffer_idx >= 0 && buffer_idx < output_buffer_len) {
                                       // Handle cases where weight might be zero but data exists (e.g., start/end)
                                       val = output_buffer(i, j, buffer_idx);
                                   }
                                   val = val * ref_std + ref_mean;
                                   processed_chunk(i, j, k) = val;
                               }
                          }
                      }
                      // --- Call the Callback ---
                      write_callback(processed_chunk, actual_samples_this_block);
                 }
             } // end if original indices valid

             // --- Shift the Rolling Buffers ---
             if (num_samples_to_process < output_buffer_len) {
                 Eigen::TensorMap<Eigen::Tensor<float, 3>> buf_map(output_buffer.data(), nb_out_sources, num_channels, output_buffer_len);
                 Eigen::TensorMap<Eigen::Tensor<float, 1>> sw_map(sum_weight_buffer.data(), output_buffer_len);
                 int remaining_samples = output_buffer_len - num_samples_to_process;
                 // Define slices
                 Eigen::array<Eigen::Index, 3> src_offset = {0, 0, num_samples_to_process};
                 Eigen::array<Eigen::Index, 3> dst_offset = {0, 0, 0};
                 Eigen::array<Eigen::Index, 3> extent = {nb_out_sources, num_channels, remaining_samples};
                 Eigen::array<Eigen::Index, 1> sw_src_offset = {num_samples_to_process};
                 Eigen::array<Eigen::Index, 1> sw_dst_offset = {0};
                 Eigen::array<Eigen::Index, 1> sw_extent = {remaining_samples};
                 // Shift data
                 buf_map.slice(dst_offset, extent) = buf_map.slice(src_offset, extent);
                 sw_map.slice(sw_dst_offset, sw_extent) = sw_map.slice(sw_src_offset, sw_extent);
                 // Zero out the end
                 Eigen::array<Eigen::Index, 3> zero_offset = {0, 0, remaining_samples};
                 Eigen::array<Eigen::Index, 3> zero_extent = {nb_out_sources, num_channels, num_samples_to_process};
                 buf_map.slice(zero_offset, zero_extent).setZero();
                 Eigen::array<Eigen::Index, 1> sw_zero_offset = {remaining_samples};
                 Eigen::array<Eigen::Index, 1> sw_zero_extent = {num_samples_to_process};
                 sw_map.slice(sw_zero_offset, sw_zero_extent).setZero();
             } else {
                 output_buffer.setZero();
                 sum_weight_buffer.setZero();
             }

             buffer_start_sample_idx += num_samples_to_process;
             samples_ready_to_write -= num_samples_to_process;

        } // end if samples_ready_to_write

        // --- Update Progress ---
        progress += progress_increment;
        cb(std::min(1.0f, progress), "Processing segment");

    } // End of segment loop


    // --- Handle Remaining Samples in the Buffer ---
    cb(1.0f, "Processing remaining samples");
    long long remaining_samples_in_buffer_calc = (shifted_length - buffer_start_sample_idx);
    // Ensure we don't read past the allocated buffer length if calculations are slightly off
    remaining_samples_in_buffer_calc = std::min(remaining_samples_in_buffer_calc, (long long)output_buffer_len);

    if (remaining_samples_in_buffer_calc > 0) {
        // --- *** FIX STARTS HERE *** ---
        // Calculate original indices for the remaining block
        long long first_sample_abs_idx_to_write = buffer_start_sample_idx;
        // Use remaining_samples_in_buffer_calc to determine the last sample index
        long long last_sample_abs_idx_to_write = buffer_start_sample_idx + remaining_samples_in_buffer_calc - 1;
        long long original_first_idx = first_sample_abs_idx_to_write - (max_shift - shift_offset);
        long long original_last_idx = last_sample_abs_idx_to_write - (max_shift - shift_offset);

        int actual_samples_this_block = 0; // Declare and initialize

        // Check if any part of this remaining block falls within the original audio length
        if (original_last_idx >= 0 && original_first_idx < length) {
            long long write_start_orig = std::max(0LL, original_first_idx);
            long long write_end_orig = std::min((long long)length - 1, original_last_idx);
            actual_samples_this_block = static_cast<int>(write_end_orig - write_start_orig + 1);
        }
        // --- *** FIX ENDS HERE *** ---


        if (actual_samples_this_block > 0) {
            // Calculate the starting index within the rolling buffer for these samples
            // Need to use write_start_orig which corresponds to the actual start in the original audio
            int buffer_write_start_idx = static_cast<int>(std::max(0LL, original_first_idx) + (max_shift - shift_offset) - buffer_start_sample_idx);
            // Clamp start index to be non-negative relative to buffer start
            buffer_write_start_idx = std::max(0, buffer_write_start_idx);

            // Ensure we don't read past the calculated remaining samples in the buffer
            // Adjust actual_samples_this_block if buffer_write_start_idx + count exceeds remaining_samples_in_buffer_calc
            if (buffer_write_start_idx + actual_samples_this_block > remaining_samples_in_buffer_calc) {
                 actual_samples_this_block = static_cast<int>(remaining_samples_in_buffer_calc - buffer_write_start_idx);
                 // Ensure it's not negative if start index was already at the end
                 actual_samples_this_block = std::max(0, actual_samples_this_block);
            }


             if (actual_samples_this_block > 0) {
                 // Prepare final chunk
                 if (processed_chunk.dimension(2) != actual_samples_this_block) {
                     processed_chunk.resize(nb_out_sources, num_channels, actual_samples_this_block);
                 }

                 // Normalize and Denormalize into processed_chunk
                 for (int i = 0; i < nb_out_sources; ++i) {
                     for (int j = 0; j < num_channels; ++j) {
                         for (int k = 0; k < actual_samples_this_block; ++k) {
                             int buffer_idx = buffer_write_start_idx + k;
                             float val = 0.0f;
                             // Check buffer bounds before accessing
                             if (buffer_idx >= 0 && buffer_idx < output_buffer_len) {
                                 if (buffer_idx < sum_weight_buffer.size() && sum_weight_buffer(buffer_idx) > 1e-8f) {
                                     val = output_buffer(i, j, buffer_idx) / sum_weight_buffer(buffer_idx);
                                 } else {
                                     // Handle cases where weight might be zero but data exists
                                     val = output_buffer(i, j, buffer_idx);
                                 }
                             } // else: val remains 0.0f if buffer_idx is out of bounds

                             val = val * ref_std + ref_mean;
                             processed_chunk(i, j, k) = val;
                         }
                     }
                 }
                 // --- Call the Callback for the final chunk ---
                 write_callback(processed_chunk, actual_samples_this_block);
            }
        }
    }

    cb(1.0f, "Incremental inference finished");
}

