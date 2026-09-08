#pragma once

#include <chrono>
#include <fstream>
#include <string>

struct llama_context;

class memory_stats_writer {
public:
    memory_stats_writer(bool enabled, std::string path);

    bool enabled() const;
    bool ready() const;
    const std::string & path() const;

    void record(const char * stage, const llama_context * ctx);

private:
    bool enabled_ = false;
    std::string path_;
    std::ofstream output_;
    std::chrono::steady_clock::time_point start_;
};
