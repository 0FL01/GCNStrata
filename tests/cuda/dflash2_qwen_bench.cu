// Stand-alone gfx906 correctness + latency runner. The NumPy export format and
// invocation are in the lab's dflash2-native/make_oracle.py and report.
#include "dflash2_qwen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace df = strata::core::dflash2_qwen;
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
struct Buffer {
    float *device = nullptr, *host = nullptr;
    explicit Buffer(size_t n) {
        check(cudaMalloc(reinterpret_cast<void**>(&device), n * sizeof(float)), "allocate tap staging device buffer");
        try { check(cudaMallocHost(reinterpret_cast<void**>(&host), n * sizeof(float)), "allocate tap staging host buffer"); }
        catch (...) { (void)cudaFree(device); throw; }
    }
    ~Buffer() { (void)cudaFreeHost(host); (void)cudaFree(device); }
    Buffer(const Buffer&) = delete;
};
struct Events {
    cudaEvent_t start = nullptr, stop = nullptr;
    Events() {
        check(cudaEventCreate(&start), "create start event");
        try { check(cudaEventCreate(&stop), "create stop event"); }
        catch (...) { (void)cudaEventDestroy(start); throw; }
    }
    ~Events() { (void)cudaEventDestroy(start); (void)cudaEventDestroy(stop); }
};

std::ifstream open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return f;
}
template <class T> void read(std::istream& f, T* p, size_t n) {
    if (!f.read(reinterpret_cast<char*>(p), n * sizeof(T))) throw std::runtime_error("short benchmark input read");
}
template <class T> T read(std::istream& f) { T x; read(f, &x, 1); return x; }
std::streamoff length(std::ifstream& f) { f.seekg(0, std::ios::end); return f.tellg(); }
constexpr size_t tap_bytes = df::kTapWidth * sizeof(float);
constexpr int batch = 128;

void append_from_file(df::Drafter& d, std::ifstream& taps, int64_t first, int64_t end, Buffer& buf) {
    taps.seekg(first * tap_bytes);
    for (int64_t p = first; p < end; p += batch) {
        const int b = int(std::min<int64_t>(batch, end - p));
        read(taps, buf.host, size_t(b) * df::kTapWidth);
        check(cudaMemcpyAsync(buf.device, buf.host, b * tap_bytes, cudaMemcpyHostToDevice, d.stream()), "upload fixture taps");
        d.context(buf.device, p, b);
        check(cudaStreamSynchronize(d.stream()), "complete fixture append");
    }
}

struct Metrics {
    int blocks = 0, exact = 0, steps = 0, correct = 0, same_sets = 0, overlap = 0;
    double hidden_rel = 0, worst_rel = 0, max_logit = 0;
    void add(const df::Result& r, const int32_t* tokens, const int32_t* ids, const float* logits,
             const float* hidden, const float* reference) {
        ++blocks;
        exact += std::equal(tokens, tokens + r.steps, r.tokens.begin());
        steps += r.steps;
        for (int t = 0; t < r.steps; ++t) {
            if (r.prob[t] != 1.f) throw std::runtime_error("wrong greedy proposal probability");
            correct += r.tokens[t] == tokens[t];
            const int base = t * df::kTop;
            const auto* row = r.top_ids.data() + base;
            if (std::find(row, row + df::kTop, r.tokens[t]) == row + df::kTop)
                throw std::runtime_error("proposal absent from top-16");
            int common = 0;
            for (int i = 0; i < df::kTop; ++i) {
                const float u = r.unary[base + i];
                if (row[i] < 0 || row[i] >= df::kVocab || !std::isfinite(u) ||
                    std::find(row, row + i, row[i]) != row + i ||
                    (i && (u > r.unary[base + i - 1] || (u == r.unary[base + i - 1] && row[i] < row[i - 1]))))
                    throw std::runtime_error("invalid candidate row");
                const auto* found = std::find(ids + base, ids + base + df::kTop, row[i]);
                if (found != ids + base + df::kTop) {
                    ++common;
                    max_logit = std::max(max_logit, std::abs(double(u) - logits[found - ids]));
                }
            }
            same_sets += common == df::kTop;
            overlap += common;
        }
        double norm = 0, err = 0;
        for (int j = 0; j < (r.steps + 1) * df::kWidth; ++j) {
            if (!std::isfinite(hidden[j]) || !std::isfinite(reference[j])) throw std::runtime_error("nonfinite hidden value");
            const double v = reference[j], diff = hidden[j] - v;
            norm += v * v;
            err += diff * diff;
        }
        const double rel = std::sqrt(err / std::max(norm, 1e-30));
        hidden_rel += rel;
        worst_rel = std::max(worst_rel, rel);
    }
    void print(const std::string& key, int b) const {
        std::cout << "PARITY " << key << " B=" << b << " exact_blocks=" << exact << '/' << blocks
                  << " exact_ids=" << correct << '/' << steps << " exact_sets=" << same_sets << '/' << steps
                  << " top16_overlap=" << overlap << '/' << steps * df::kTop
                  << " hidden_rel_mean=" << hidden_rel / blocks << " hidden_rel_max=" << worst_rel
                  << " logit_abs_max=" << max_logit << '\n';
    }
};

void validate(df::Drafter& d, const std::string& stem, const std::string& path,
              std::array<Metrics, df::kBlock + 1>& all) {
    auto f = open(path);
    auto taps = open(stem + ".taps.f32");
    auto ids = open(stem + ".ids.bin");
    const auto n = length(ids) / sizeof(int32_t);
    if (n == 0 || length(ids) != n * sizeof(int32_t) || length(taps) != n * tap_bytes)
        throw std::runtime_error("mismatched fixture lengths");
    char magic[8];
    read(f, magic, sizeof(magic));
    if (std::memcmp(magic, "Q38DF2\0\0", 8)) throw std::runtime_error("bad oracle magic");
    const unsigned count = read<uint32_t>(f);
    int64_t next = read<uint32_t>(f);
    if (!count) throw std::runtime_error("empty oracle");
    std::array<Metrics, df::kBlock + 1> metrics{};
    Buffer staged(batch * df::kTapWidth);
    d.reset();
    for (unsigned i = 0; i < count; ++i) {
        const int64_t pos = read<uint32_t>(f);
        const int b = read<uint32_t>(f), token = read<uint32_t>(f);
        if (pos < next || b < 2 || b > df::kBlock || pos + b > int64_t(n))
            throw std::runtime_error("invalid oracle position or block width");
        ids.seekg(pos * sizeof(int32_t));
        if (token != read<int32_t>(ids)) throw std::runtime_error("oracle and fixture anchors differ");
        int32_t proposals[df::kBlock - 1], top[(df::kBlock - 1) * df::kTop];
        float logit[(df::kBlock - 1) * df::kTop], expected[df::kWidth * df::kBlock], actual[df::kWidth * df::kBlock];
        read(f, proposals, b - 1);
        read(f, top, (b - 1) * df::kTop);
        read(f, logit, (b - 1) * df::kTop);
        read(f, expected, b * df::kWidth);
        if (pos > next) append_from_file(d, taps, next, pos, staged);
        next = pos;
        const auto r = d.ask(token, pos, b, nullptr, actual);
        metrics[b].add(r, proposals, top, logit, actual, expected);
        all[b].add(r, proposals, top, logit, actual, expected);
    }
    if (f.peek() != EOF) throw std::runtime_error("extra oracle data");
    const auto name = stem.substr(stem.find_last_of('/') + 1);
    for (int b = 2; b <= df::kBlock; ++b) if (metrics[b].blocks) metrics[b].print(name, b);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return (values[n / 2] + values[(n - 1) / 2]) / 2;
}

void latency(df::Drafter& d, const std::string& stem, int reps) {
    constexpr int warmups = 5;
    auto ids = open(stem + ".ids.bin"), start = open(stem + ".start"), taps = open(stem + ".taps.f32");
    int64_t p = 0;
    const auto n = length(ids) / sizeof(int32_t);
    if (!(start >> p) || p < df::kWindow || p + 1 >= int64_t(n) || length(taps) != n * tap_bytes)
        throw std::runtime_error("latency fixture must have a full 2048-row prefix and a non-padding anchor tap");
    ids.seekg(p * sizeof(int32_t));
    const auto token = read<int32_t>(ids);
    Buffer buf(batch * df::kTapWidth);
    d.reset();
    append_from_file(d, taps, p - df::kWindow, p, buf);
    std::cout << "LATENCY fixture=" << stem << " pos=" << p << " ring=" << df::kWindow
              << " window=trtllm reps=" << reps << " warmups=" << warmups << '\n';
    for (int b : {2, 4, 6, 8}) {
        for (int i = 0; i < warmups; ++i) (void)d.ask(token, p, b);
        std::vector<double> layers, head, selector, total, wall;
        for (int i = 0; i < reps; ++i) {
            df::Timing t;
            const auto begin = std::chrono::steady_clock::now();
            (void)d.ask(token, p, b, &t);
            wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
            layers.push_back(t.layers_ms); head.push_back(t.head_ms);
            selector.push_back(t.selector_ms); total.push_back(t.total_ms);
        }
        std::cout << "ASK B=" << b << " gpu_ms=" << median(total) << " layers_ms=" << median(layers)
                  << " head_ms=" << median(head) << " selector_ms=" << median(selector)
                  << " wall_ms=" << median(wall) << '\n';
    }
    // Keep input GPU-resident and use the same 128 real tap rows for every variant.
    const int64_t src = std::min<int64_t>(p, n - batch);
    taps.seekg(src * tap_bytes);
    read(taps, buf.host, batch * df::kTapWidth);
    check(cudaMemcpyAsync(buf.device, buf.host, batch * tap_bytes, cudaMemcpyHostToDevice, d.stream()), "stage context timing rows");
    Events e;
    for (int rows : {1, 8, batch}) {
        for (int i = 0; i < warmups; ++i) { d.context(buf.device, p, rows); p += rows; }
        std::vector<double> times;
        for (int i = 0; i < reps; ++i) {
            check(cudaEventRecord(e.start, d.stream()), "record context start");
            d.context(buf.device, p, rows); p += rows;
            check(cudaEventRecord(e.stop, d.stream()), "record context stop");
            check(cudaEventSynchronize(e.stop), "complete timed context append");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, e.start, e.stop), "time context append");
            times.push_back(ms);
        }
        std::cout << "CONTEXT rows_per_call=" << rows << " gpu_ms=" << median(times)
                  << " gpu_ms_per_row=" << median(times) / rows << '\n';
    }
}

using Histogram = std::array<std::array<int, df::kBlock>, df::kBlock + 1>;
void acceptance(df::Drafter& d, const std::string& stem, Histogram& all) {
    auto ids_file = open(stem + ".ids.bin"), start = open(stem + ".start"), taps = open(stem + ".taps.f32");
    const auto bytes = length(ids_file);
    const int64_t n = bytes / sizeof(int32_t);
    int64_t p = 0;
    if (bytes % sizeof(int32_t) || n < 2 || !(start >> p) || p < 0 || p >= n - 1 || length(taps) != n * tap_bytes)
        throw std::runtime_error("invalid acceptance trajectory");
    ids_file.seekg(0);
    std::vector<int32_t> ids(n);
    read(ids_file, ids.data(), n);
    if (std::any_of(ids.begin(), ids.end(), [](int id) { return id < 0 || id >= df::kVocab; }))
        throw std::runtime_error("invalid trajectory token ID");
    Buffer staged(batch * df::kTapWidth);
    d.reset();
    append_from_file(d, taps, std::max<int64_t>(0, p - df::kWindow + 1), p, staged);
    Histogram hist{};
    const int64_t stop = std::min<int64_t>(p + 160, n - 1);
    for (; p < stop; ++p) {
        for (int b : {2, 4, 8}) {
            if (p + b > n) continue;  // Same width-specific stops as accept_study.py.
            const auto res = d.ask(ids[p], p, b);
            int accepted = 0;
            while (accepted < b - 1 && res.tokens[accepted] == ids[p + accepted + 1]) ++accepted;
            ++hist[b][accepted];
            ++all[b][accepted];
        }
        if (p + 1 < stop) append_from_file(d, taps, p, p + 1, staged);
    }
    const auto name = stem.substr(stem.find_last_of('/') + 1);
    for (int b : {2, 4, 8}) {
        int count = 0, sum = 0;
        for (int i = 0; i < b; ++i) { count += hist[b][i]; sum += i * hist[b][i]; }
        std::cout << "ACCEPT " << name << " B=" << b << " N=" << count << " sum=" << sum
                  << " mean=" << (count ? double(sum) / count : 0) << " hist=";
        for (int i = 0; i < b; ++i) std::cout << (i ? "," : "") << hist[b][i];
        std::cout << std::endl;
    }
}
}  // namespace

int main(int argc, char** argv) try {
    const bool val = argc >= 4 && std::string(argv[3]) == "validate";
    const bool bench = argc == 6 && std::string(argv[3]) == "latency";
    const bool study = argc >= 5 && std::string(argv[3]) == "accept";
    if ((!val || argc < 6 || argc % 2) && !bench && !study) {
        std::cerr << "usage: " << argv[0] << " DRAFT_GGUF TARGET_SHARD validate TAP_STEM ORACLE [TAP_STEM ORACLE ...]\n"
                  << "       " << argv[0] << " DRAFT_GGUF TARGET_SHARD latency TAP_STEM REPS\n"
                  << "       " << argv[0] << " DRAFT_GGUF TARGET_SHARD accept TAP_STEM [TAP_STEM ...]\n";
        return 2;
    }
    check(cudaSetDevice(0), "set benchmark device 0");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "read GPU properties");
    std::cout << "DEVICE " << props.name << " arch=" << props.gcnArchName << " wave=" << props.warpSize
              << " draft=" << argv[1] << '\n' << std::setprecision(8);
    size_t free_before = 0, free_after = 0, mem_total = 0;
    check(cudaMemGetInfo(&free_before, &mem_total), "read free VRAM");
    df::Drafter model(argv[1], argv[2]);
    check(cudaMemGetInfo(&free_after, &mem_total), "read initialized free VRAM");
    std::cout << "VRAM init_delta_bytes=" << (int64_t(free_before) - int64_t(free_after)) << '\n';
    if (study) {
        Histogram all{};
        for (int i = 4; i < argc; ++i) acceptance(model, argv[i], all);
        for (int b : {2, 4, 8}) {
            int count = 0, sum = 0;
            for (int i = 0; i < b; ++i) { count += all[b][i]; sum += i * all[b][i]; }
            std::cout << "ACCEPT ALL B=" << b << " N=" << count << " sum=" << sum
                      << " mean=" << (count ? double(sum) / count : 0) << " hist=";
            for (int i = 0; i < b; ++i) std::cout << (i ? "," : "") << all[b][i];
            std::cout << '\n';
        }
        return 0;
    }
    if (val) {
        std::array<Metrics, df::kBlock + 1> all{};
        for (int i = 4; i < argc; i += 2) validate(model, argv[i], argv[i + 1], all);
        bool pass = true;
        for (int b = 2; b <= df::kBlock; ++b) {
            const auto& m = all[b];
            if (!m.blocks) continue;
            m.print("ALL", b);
            pass &= m.exact >= 0.9 * m.blocks && m.overlap >= 0.98 * m.steps * df::kTop && m.worst_rel <= 0.01;
        }
        std::cout << (pass ? "PASS" : "FAIL") << " (>=90% exact blocks, >=98% top-16 overlap, <=1% max hidden relative L2)\n";
        return pass ? 0 : 1;
    }
    size_t parsed = 0;
    const int reps = std::stoi(argv[5], &parsed);
    if (parsed != std::strlen(argv[5]) || reps < 1 || reps > 10000) throw std::invalid_argument("REPS must be in 1..10000");
    latency(model, argv[4], reps);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "dflash2_qwen_bench: " << e.what() << '\n';
    return 1;
}
