#include "wav_writer.hpp"
#include <iostream>
#include <vector> // Include vector for temporary buffer if needed

// Helper function for writing little-endian values
inline void write_le_uint16(std::ofstream& stream, uint16_t value) {
    stream.put(static_cast<char>(value & 0xFF));
    stream.put(static_cast<char>((value >> 8) & 0xFF));
}

inline void write_le_uint32(std::ofstream& stream, uint32_t value) {
    stream.put(static_cast<char>(value & 0xFF));
    stream.put(static_cast<char>((value >> 8) & 0xFF));
    stream.put(static_cast<char>((value >> 16) & 0xFF));
    stream.put(static_cast<char>((value >> 24) & 0xFF));
}


StreamingWavWriter::StreamingWavWriter()
    : sample_rate(0), num_channels(0), bytes_per_sample(0),
      block_align(0), data_chunk_size(0), is_open(false),
      riff_chunk_size_pos(0), data_chunk_size_pos(0) {}

StreamingWavWriter::~StreamingWavWriter() {
    if (is_open) {
        finalize(); // Attempt to finalize if not already done
    }
}

bool StreamingWavWriter::open(const std::string& filename, uint32_t sr, uint16_t nc) {
    if (is_open) {
        std::cerr << "WAV Writer Error: File already open: " << current_filename << std::endl;
        return false;
    }
    if (nc != 1 && nc != 2) {
         std::cerr << "WAV Writer Error: Only 1 or 2 channels supported. Requested: " << nc << std::endl;
         return false;
    }

    sample_rate = sr;
    num_channels = nc;
    // Using 32-bit float PCM
    bytes_per_sample = sizeof(float);
    block_align = num_channels * bytes_per_sample;
    data_chunk_size = 0; // Start with zero data

    file_stream.open(filename, std::ios::binary | std::ios::trunc);
    if (!file_stream) {
        std::cerr << "WAV Writer Error: Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    current_filename = filename;
    write_header(); // Write header with placeholders
    is_open = file_stream.good();

    if (!is_open) {
         std::cerr << "WAV Writer Error: Stream state bad after writing header for: " << filename << std::endl;
         file_stream.close(); // Ensure closed on error
    }

    return is_open;
}

void StreamingWavWriter::write_header() {
    // RIFF Chunk Descriptor
    file_stream.write("RIFF", 4);
    riff_chunk_size_pos = file_stream.tellp();
    write_le_uint32(file_stream, 0); // Placeholder for ChunkSize (file size - 8)
    file_stream.write("WAVE", 4);

    // "fmt " sub-chunk
    file_stream.write("fmt ", 4);
    write_le_uint32(file_stream, 16); // Subchunk1Size for PCM (16 bytes)
    // AudioFormat (3 for IEEE float)
    uint16_t audio_format = 3;
    write_le_uint16(file_stream, audio_format);
    write_le_uint16(file_stream, num_channels);
    write_le_uint32(file_stream, sample_rate);
    // ByteRate == SampleRate * NumChannels * BytesPerSample
    write_le_uint32(file_stream, sample_rate * block_align);
    write_le_uint16(file_stream, block_align);
    // BitsPerSample == BytesPerSample * 8
    write_le_uint16(file_stream, bytes_per_sample * 8);

    // "data" sub-chunk
    file_stream.write("data", 4);
    data_chunk_size_pos = file_stream.tellp();
    write_le_uint32(file_stream, 0); // Placeholder for Subchunk2Size (data size)
}


bool StreamingWavWriter::append_samples(const float* interleaved_data, size_t num_total_samples) {
     if (!is_open) return false;
     if (num_total_samples == 0) return true; // Nothing to write

     size_t bytes_to_write = num_total_samples * bytes_per_sample;
     file_stream.write(reinterpret_cast<const char*>(interleaved_data), bytes_to_write);

     if (!file_stream) {
          std::cerr << "WAV Writer Error: Failed to write sample data to: " << current_filename << std::endl;
          is_open = false; // Mark as failed
          return false;
     }

     data_chunk_size += static_cast<uint32_t>(bytes_to_write);
     return true;
}


bool StreamingWavWriter::append_samples(const float* chan0_data, const float* chan1_data, size_t num_samples_per_channel) {
    if (!is_open) return false;
    if (num_samples_per_channel == 0) return true;
    if (num_channels != 2) {
         std::cerr << "WAV Writer Error: append_samples(chan0, chan1, ...) called on non-stereo writer." << std::endl;
         return false;
    }

    // Interleave data into a temporary buffer before writing
    // This uses temporary RAM but avoids many small writes. Adjust buffer size if needed.
    const size_t buffer_chunk_size = 4096; // Process N samples at a time
    std::vector<float> interleaved_buffer(buffer_chunk_size * num_channels);

    size_t samples_written = 0;
    while(samples_written < num_samples_per_channel) {
        size_t samples_to_process = std::min(buffer_chunk_size, num_samples_per_channel - samples_written);
        for (size_t i = 0; i < samples_to_process; ++i) {
            interleaved_buffer[i * 2]     = chan0_data[samples_written + i];
            interleaved_buffer[i * 2 + 1] = chan1_data[samples_written + i];
        }

        size_t bytes_to_write = samples_to_process * block_align;
        file_stream.write(reinterpret_cast<const char*>(interleaved_buffer.data()), bytes_to_write);

        if (!file_stream) {
             std::cerr << "WAV Writer Error: Failed to write interleaved sample data to: " << current_filename << std::endl;
             is_open = false; // Mark as failed
             return false;
        }

        data_chunk_size += static_cast<uint32_t>(bytes_to_write);
        samples_written += samples_to_process;
    }

    return true;
}


bool StreamingWavWriter::finalize() {
    if (!is_open) {
        // Allow finalize to be called multiple times, but only act once
        // Or if it failed previously, don't try again.
        return file_stream.is_open(); // Return true if it was successfully closed before
    }

    // Get current position = end of data = total file size
    std::streampos end_pos = file_stream.tellp();

    // Calculate final sizes
    // RIFF Chunk Size = FileSize - 8 bytes (for "RIFF" and the size field itself)
    uint32_t riff_chunk_size = static_cast<uint32_t>(end_pos) - 8;
    // Data Chunk Size is already tracked in data_chunk_size

    // Seek back and write the final sizes
    file_stream.seekp(riff_chunk_size_pos);
    if (!file_stream) { std::cerr << "WAV Writer Error: Seek failed (RIFF size) for " << current_filename << std::endl; return false; }
    write_le_uint32(file_stream, riff_chunk_size);

    file_stream.seekp(data_chunk_size_pos);
    if (!file_stream) { std::cerr << "WAV Writer Error: Seek failed (data size) for " << current_filename << std::endl; return false; }
    write_le_uint32(file_stream, data_chunk_size);

    // Seek back to the end of the file before closing (optional but good practice)
    file_stream.seekp(end_pos);
    file_stream.close();

    is_open = false; // Mark as closed
    bool success = !file_stream.fail();
    if(success) {
        std::cout << "Finalized WAV file: " << current_filename << " (" << data_chunk_size << " data bytes)" << std::endl;
    } else {
        std::cerr << "WAV Writer Error: Stream failed during finalization for " << current_filename << std::endl;
    }
    return success;
}

size_t StreamingWavWriter::get_total_samples_written() const {
    if (block_align == 0) return 0;
    return data_chunk_size / block_align;
}
