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

// Note: Random generator is not used within process_chunk as shift is pre-calculated
// static std::random_device rd_apply;
// static std::mt19937 gen_apply(rd_apply());

namespace demucsonnx {

// --- Padding Helper (remains the same, ensure it handles edge cases) ---
static void reflect_padding(
                            Eigen::MatrixXf &padded_mix, // The buffer to pad (includes original data + padding space)
                            int data_start_in_padded,    // Index where the actual data starts in padded_mix
                            int data_length,             // Length of the actual data within padded_mix
                            int total_padded_length)     // Total size of the padded_mix buffer
{
    const int num_channels = padded_mix.rows();
    const int left_padding_count = data_start_in_padded;
    const int right_padding_count = total_padded_length - (data_start_in_padded + data_length);
    const int data_end_in_padded = data_start_in_padded + data_length -1;

    // Reflect left
    for (int i = 0; i < left_padding_count; ++i) {
        int src_idx = data_start_in_padded + i; // Source index within the actual data part
        int dst_idx = data_start_in_padded - 1 - i; // Target index in the left padding area

        // Boundary checks
        if (dst_idx >= 0 && src_idx < (data_start_in_padded + data_length)) {
            padded_mix.block(0, dst_idx, num_channels, 1) = padded_mix.block(0, src_idx, num_channels, 1);
        } else if (dst_idx >= 0) {
            // Handle cases where padding is larger than data: fill with edge or zero
             padded_mix.block(0, dst_idx, num_channels, 1) = padded_mix.block(0, data_start_in_padded, num_channels, 1); // Use first data sample
            // padded_mix.block(0, dst_idx, num_channels, 1).setZero(); // Alternative: fill with zero
        }
    }

    // Reflect right
    for (int i = 0; i < right_padding_count; ++i) {
        int src_idx = data_end_in_padded - i; // Source index within the actual data part
        int dst_idx = data_end_in_padded + 1 + i; // Target index in the right padding area

        // Boundary checks
        if (dst_idx < total_padded_length && src_idx >= data_start_in_padded) {
            padded_mix.block(0, dst_idx, num_channels, 1) = padded_mix.block(0, src_idx, num_channels, 1);
        } else if (dst_idx < total_padded_length) {
            // Handle cases where padding is larger than data: fill with edge or zero
             padded_mix.block(0, dst_idx, num_channels, 1) = padded_mix.block(0, data_end_in_padded, num_channels, 1); // Use last data sample
            // padded_mix.block(0, dst_idx, num_channels, 1).setZero(); // Alternative: fill with zero
        }
    }
}


DemucsResultCode demucs_inference_process_chunk(
    demucsonnx::demucs_model &model,
    const Eigen::MatrixXf &audio_chunk,         // normalized audio for this chunk (now monotonic start frame)
    long long chunk_start_frame_shifted,        // global start frame of this chunk (now monotonic)
    Eigen::Tensor<float, 3>& output_buffer,     // overlap-add output buffer (circular)
    demucsonnx::demucs_segment_buffers& buffers,// reusable buffers for segment processing
    demucsonnx::stft_buffers& stft_buf,         // reusable STFT buffers
    const std::atomic<bool>* cancel_flag
) {
    const int chunk_length    = audio_chunk.cols();
    const int nb_out_sources  = model.nb_sources;
    const int num_channels    = 2;
    // REMOVED: segment_samples based on weight - use buffers.segment_samples instead if needed
    // const int segment_samples = weight ? weight->size() : buffers.segment_samples;
    const int segment_samples = buffers.segment_samples; // Length of processing segment expected by model_inference buffers
    // REMOVED: stride_samples calculation - not needed here
    // const int stride_samples  = static_cast<int>((1.0f - demucsonnx::OVERLAP) * segment_samples);
    const int stride_samples = static_cast<int>((1.0f - demucsonnx::OVERLAP) * segment_samples); // Need stride for looping
    const int output_buffer_len = output_buffer.dimension(2);

    if (chunk_length == 0) {
        return DEMUCS_RESULT_SUCCESS;  // nothing to do
    }

    // Iterate over segment-sized windows within this chunk
    // Loop should go up to chunk_length - stride_samples? No, needs to cover the full chunk.
    // The last segment might be shorter.
    for (long long segment_offset_in_chunk = 0;
         segment_offset_in_chunk < chunk_length; // Process segments starting up to the last possible position
         segment_offset_in_chunk += stride_samples)
    {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            return DEMUCS_RESULT_CANCELLED;
        }

        // Determine current segment length (might be shorter at end of chunk)
        // This is the amount of *real* data we extract for this segment
        long long current_segment_data_length = std::min<long long>(segment_samples, chunk_length - segment_offset_in_chunk);
        if (current_segment_data_length <= 0) continue; // Should not happen with loop condition? Safety check.


        // Extract the segment data from the input chunk
        Eigen::Ref<const Eigen::MatrixXf> segment_chunk_data =
            audio_chunk.block(0, segment_offset_in_chunk, num_channels, (int) current_segment_data_length);

        // Padding is handled relative to the *model's* expected segment size (buffers.segment_samples)
        // Compute symmetric padding amounts needed to reach buffers.segment_samples
        const int pad_total = buffers.segment_samples - (int) current_segment_data_length;
         // Ensure padding is non-negative
         const int actual_pad_total = std::max(0, pad_total);
        const int symmetric_padding_start = actual_pad_total / 2;


        /* ----- Prepare padded input for the model ----- */
        buffers.padded_mix.setZero();
        // Place the actual segment data in the center of the *model's* required segment length, offset by internal pad
        buffers.padded_mix.block(0, buffers.pad + symmetric_padding_start,
                                 num_channels, (int) current_segment_data_length) = segment_chunk_data;
        // Apply reflection padding based on the actual data placed and the model's segment size
        reflect_padding(buffers.padded_mix,
                        buffers.pad + symmetric_padding_start, // Start of actual data in padded_mix
                        (int) current_segment_data_length,     // Length of actual data
                        buffers.padded_segment_samples);       // Total size expected by model (incl. internal pad)


        /* ----- Run the Demucs model on this padded segment ----- */
        try {
            // model_inference uses buffers.padded_mix and outputs to buffers.targets_out
            demucsonnx::model_inference(model, buffers, stft_buf);
        } catch (const std::exception& e) {
            std::cerr << "Error [demucs_inference_process_chunk]: Exception during model_inference: "
                      << e.what() << std::endl;
            return DEMUCS_RESULT_ERROR_INFERENCE;
        } catch (...) {
            std::cerr << "Error [demucs_inference_process_chunk]: Unknown exception during model_inference."
                      << std::endl;
            return DEMUCS_RESULT_ERROR_INFERENCE;
        }

        // Calculate the global start frame of this segment (based on monotonic input start frame)
        long long segment_start_frame_global = chunk_start_frame_shifted + segment_offset_in_chunk;

        /* ----- Overlap-add: accumulate the model output into the circular buffer ----- */
        // The output buffers.targets_out contains the processed audio corresponding
        // to the *model's segment length* (buffers.segment_samples).
        // We need to add the portion corresponding to the actual data length + fade-out.
        // The length to add here should correspond to the original segment length used in OLA (seg)
        const int ola_segment_length = buffers.segment_samples; // Corresponds to the length processed by istft

        for (int src = 0; src < nb_out_sources; ++src) {
            for (int ch = 0; ch < num_channels; ++ch) {
                // Add the relevant portion of the processed segment to the OLA buffer
                for (int k = 0; k < ola_segment_length; ++k) {
                    // Global frame index for this sample
                    long long global_idx = segment_start_frame_global + k;
                    long long buffer_idx = global_idx % output_buffer_len;
                    if (buffer_idx < 0) buffer_idx += output_buffer_len;

                    // Index within the model's output tensor (buffers.targets_out)
                    // This assumes buffers.targets_out contains `ola_segment_length` valid samples
                    int target_out_idx = k;

                    // Check bounds before adding
                    if (target_out_idx >= 0 && target_out_idx < buffers.targets_out.dimension(2)) {
                         // Directly add the output (no secondary windowing)
                         output_buffer(src, ch, (int) buffer_idx) += buffers.targets_out(src, ch, target_out_idx);
                    } else if (target_out_idx >= buffers.targets_out.dimension(2)) {
                        // If k goes beyond the model output size, stop adding for this segment/channel/source
                        break;
                    }
                     // REMOVED: Update of sum_weight_buffer
                }
            }
        }
    }
    return DEMUCS_RESULT_SUCCESS;
}
} // namespace demucsonnx
