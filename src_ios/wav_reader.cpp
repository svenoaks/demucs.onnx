// wav_reader.cpp
#include "wav_reader.hpp"
#include <iostream>
#include <stdexcept> // For std::runtime_error

StreamingWavReader::StreamingWavReader() {
    // Initialize drwav structure to zeros
    // drwav_init* functions will handle proper initialization later
    memset(&wav, 0, sizeof(drwav));
}

StreamingWavReader::~StreamingWavReader() {
    close();
}

bool StreamingWavReader::open(const std::string& filename) {
    close(); // Close any previously opened file

    if (!drwav_init_file(&wav, filename.c_str(), NULL)) {
        std::cerr << "Error [WavReader]: Failed to initialize WAV file: " << filename << std::endl;
        is_file_open = false;
        return false;
    }

    // --- Validate Format ---
    if (wav.sampleRate != 44100) { // Check against demucsonnx::SUPPORTED_SAMPLE_RATE if available
         std::cerr << "Error [WavReader]: Unsupported sample rate: " << wav.sampleRate << " (expected 44100)" << std::endl;
         drwav_uninit(&wav);
         is_file_open = false;
         return false;
    }
     if (wav.channels != 2) { // Only stereo supported by current pipeline
         std::cerr << "Error [WavReader]: Unsupported channel count: " << wav.channels << " (expected 2)" << std::endl;
         drwav_uninit(&wav);
         is_file_open = false;
         return false;
     }
    // We expect float input for the model pipeline
    if (wav.translatedFormatTag != DR_WAVE_FORMAT_IEEE_FLOAT || wav.bitsPerSample != 32) {
         std::cerr << "Warning [WavReader]: Input format is not 32-bit float. Conversion will occur during read." << std::endl;
         // Allow proceeding, as drwav_read_pcm_frames_f32 handles conversion
    }


    sample_rate = wav.sampleRate;
    channels = wav.channels;
    total_pcm_frames = wav.totalPCMFrameCount;
    current_filename = filename;
    is_file_open = true;

    std::cout << "Info [WavReader]: Opened '" << filename << "', Rate: " << sample_rate
              << ", Channels: " << channels << ", Frames: " << total_pcm_frames << std::endl;

    return true;
}

drwav_uint64 StreamingWavReader::read_chunk(Eigen::MatrixXf& buffer, drwav_uint64 max_frames_to_read) {
    if (!is_file_open || max_frames_to_read == 0) {
        return 0;
    }

    // Ensure internal buffer is large enough
    size_t required_interleaved_samples = max_frames_to_read * channels;
    if (read_buffer_interleaved.size() < required_interleaved_samples) {
        try {
             read_buffer_interleaved.resize(required_interleaved_samples);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Error [WavReader]: Failed to allocate read buffer: " << e.what() << std::endl;
            return 0; // Indicate error / no frames read
        }
    }

    // Read interleaved float data using dr_wav
    drwav_uint64 frames_read = drwav_read_pcm_frames_f32(
        &wav,
        max_frames_to_read,
        read_buffer_interleaved.data()
    );

    if (frames_read == 0) {
        // End of file or error
        return 0;
    }

    // Resize Eigen buffer if needed (common case: last chunk is smaller)
    // Only resize if the current size doesn't match frames_read
    if (static_cast<drwav_uint64>(buffer.cols()) != frames_read) {
         try {
            buffer.resize(channels, frames_read);
         } catch (const std::bad_alloc& e) {
             std::cerr << "Error [WavReader]: Failed to resize Eigen buffer: " << e.what() << std::endl;
             // We read the data into the internal buffer, but can't put it in Eigen.
             // Return 0 to signal an issue preventing further processing of this chunk.
             // Seek back so the caller can potentially retry with a smaller chunk if applicable?
             // For now, just return 0. The internal drwav cursor has advanced.
             seek_to_frame(wav.readCursorInPCMFrames - frames_read); // Attempt to rewind
             return 0;
         }
    }


    // Deinterleave into the Eigen::MatrixXf buffer
    // Assumes buffer has correct dimensions (channels x frames_read)
    for (drwav_uint64 i = 0; i < frames_read; ++i) {
        // TODO: Check channel count consistency? Already checked in open().
        buffer(0, i) = read_buffer_interleaved[i * channels + 0]; // Left
        buffer(1, i) = read_buffer_interleaved[i * channels + 1]; // Right
    }

    return frames_read;
}


bool StreamingWavReader::seek_to_frame(drwav_uint64 frame_index) {
    if (!is_file_open) {
        return false;
    }
    return drwav_seek_to_pcm_frame(&wav, frame_index);
}

uint32_t StreamingWavReader::get_sample_rate() const {
    return sample_rate;
}

uint16_t StreamingWavReader::get_channels() const {
    return channels;
}

drwav_uint64 StreamingWavReader::get_total_frames() const {
    return total_pcm_frames;
}

bool StreamingWavReader::is_open() const {
     return is_file_open;
}

void StreamingWavReader::close() {
    if (is_file_open) {
        drwav_uninit(&wav);
        is_file_open = false;
        sample_rate = 0;
        channels = 0;
        total_pcm_frames = 0;
        current_filename = "";
        // Optionally clear internal buffer to release memory
        // read_buffer_interleaved.clear();
        // read_buffer_interleaved.shrink_to_fit();
        std::cout << "Info [WavReader]: Closed file." << std::endl;
    }
     // Ensure wav struct is zeroed even if never opened successfully
     memset(&wav, 0, sizeof(drwav));
}