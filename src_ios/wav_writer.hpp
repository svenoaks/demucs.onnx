#ifndef WAV_WRITER_HPP
#define WAV_WRITER_HPP

#include <string>
#include <fstream>
#include <vector>
#include <cstdint> // For uint16_t, uint32_t

class StreamingWavWriter {
public:
    StreamingWavWriter();
    ~StreamingWavWriter(); // Ensure file is finalized even if not explicitly called

    // Opens the file, writes the header with placeholders
    bool open(const std::string& filename, uint32_t sample_rate, uint16_t num_channels);

    // Appends interleaved audio data
    bool append_samples(const float* interleaved_data, size_t num_samples);

    // Appends separate channel data
    bool append_samples(const float* chan0_data, const float* chan1_data, size_t num_samples_per_channel);

    // Updates the header with correct sizes and closes the file
    bool finalize();

    // Returns the total number of samples (per channel) written so far
    size_t get_total_samples_written() const;


private:
    void write_header();
    void write_uint16(uint16_t value);
    void write_uint32(uint32_t value);
    template <typename T>
    void write_bytes(const T* data, size_t count);

    std::ofstream file_stream;
    uint32_t sample_rate;
    uint16_t num_channels;
    uint16_t bytes_per_sample; // e.g., 4 for float
    uint16_t block_align;      // num_channels * bytes_per_sample
    uint32_t data_chunk_size;  // Total size of PCM data in bytes
    bool is_open;
    std::string current_filename;

    // Positions in the file where sizes need to be updated
    std::streampos riff_chunk_size_pos;
    std::streampos data_chunk_size_pos;
};

#endif // WAV_WRITER_HPP
