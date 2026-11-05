#include "BCH.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
std::size_t checks = 0;

void check(bool condition, const char* expression, int line) {
    ++checks;
    if (!condition) {
        throw std::runtime_error("line " + std::to_string(line) + ": " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void invalid(const std::function<void()>& action) {
    bool thrown = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

void flip(BCH::Bytes& data, std::size_t position) {
    data.at(position / 8) ^= static_cast<unsigned char>(1U << (7 - position % 8));
}

BCH::Bytes random_data(std::mt19937& random) {
    BCH::Bytes data(BCH::input_bytes);
    for (auto& value : data) {
        value = static_cast<unsigned char>(random() & 255U);
    }
    return data;
}

std::vector<std::size_t> error_positions(std::mt19937& random, int count) {
    std::vector<std::size_t> positions(BCH::input_bits);
    std::iota(positions.begin(), positions.end(), 0U);
    std::shuffle(positions.begin(), positions.end(), random);
    positions.resize(static_cast<std::size_t>(count));
    std::sort(positions.begin(), positions.end());
    return positions;
}

// Independent arithmetic oracle: polynomial shift/XOR, no logarithm tables.
int polynomial_multiply(int a, int b) {
    int result = 0;
    while (b != 0) {
        if ((b & 1) != 0) {
            result ^= a;
        }
        b >>= 1;
        a <<= 1;
        if ((a & 0x400) != 0) {
            a ^= 0x409;
        }
    }
    return result;
}

int polynomial_power(int base, int exponent) {
    int result = 1;
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = polynomial_multiply(result, base);
        }
        base = polynomial_multiply(base, base);
        exponent >>= 1;
    }
    return result;
}

BCH::Syndromes reference_syndromes(const BCH::Bytes& data, int t) {
    BCH::Syndromes result(static_cast<std::size_t>(2 * t + 1), 0);
    for (int root = 1; root <= 2 * t; ++root) {
        const int alpha = polynomial_power(2, root);
        int coefficient = 1;
        for (std::size_t i = 0; i < BCH::input_bits; ++i) {
            if (((data[i / 8] >> (7 - i % 8)) & 1U) != 0) {
                result[root] ^= coefficient;
            }
            coefficient = polynomial_multiply(coefficient, alpha);
        }
    }
    return result;
}

void expect_recovery(const BCH& bch, const BCH::Bytes& enrolled,
                     const std::vector<std::size_t>& positions) {
    auto noisy = enrolled;
    for (const auto position : positions) {
        flip(noisy, position);
    }
    const auto result = bch.recover(noisy, bch.compute_syndrome(enrolled));
    CHECK(result.decoded());
    CHECK(result.data == enrolled);
    CHECK(result.corrected_positions == positions);
    CHECK(result.status == (positions.empty() ? BCH::DecodeStatus::NoChange
                                             : BCH::DecodeStatus::Corrected));
}

void test_all_positions_and_random_errors() {
    std::mt19937 random(0x512BCAU);
    const BCH bch;
    const auto enrolled = random_data(random);
    for (std::size_t i = 0; i < BCH::input_bits; ++i) {
        expect_recovery(bch, enrolled, {i});
    }
    expect_recovery(bch, enrolled, {0, 1, 254, 255, 256, 257, 510, 511});
    for (int count = 0; count <= 8; ++count) {
        for (int repetition = 0; repetition < 40; ++repetition) {
            expect_recovery(bch, random_data(random), error_positions(random, count));
        }
    }
    for (const unsigned char pattern : BCH::Bytes{0x00, 0xFF, 0x55, 0xAA}) {
        expect_recovery(bch, BCH::Bytes(BCH::input_bytes, pattern), {});
        expect_recovery(bch, BCH::Bytes(BCH::input_bytes, pattern), {255, 256, 511});
    }
    for (const int t : {1, 2, 3, 4, 16, 32, 255}) {
        const BCH configured(t);
        for (const int count : {0, 1, t}) {
            expect_recovery(configured, random_data(random), error_positions(random, count));
        }
    }
}

void test_independent_syndromes_and_bit_packing() {
    std::mt19937 random(0xC0DEC0DEU);
    const BCH bch;
    for (int iteration = 0; iteration < 12; ++iteration) {
        const auto data = random_data(random);
        CHECK(bch.compute_syndrome(data) == reference_syndromes(data, 8));
        CHECK(BCH::from_bitstring(BCH::to_bitstring(data)) == data);
    }
    BCH::Bytes only_first(BCH::input_bytes, 0);
    flip(only_first, 0);
    BCH::Syndromes known(17, 1);
    known[0] = 0;
    CHECK(bch.compute_syndrome(only_first) == known);

    std::string bits(512, '0');
    bits[0] = bits[255] = bits[256] = bits[511] = '1';
    const auto bytes = BCH::from_bitstring(bits);
    CHECK(bytes[0] == 0x80);
    CHECK(bytes[31] == 0x01);
    CHECK(bytes[32] == 0x80);
    CHECK(bytes[63] == 0x01);
    CHECK(BCH::to_bitstring(bytes) == bits);
}

void test_helper_serialization() {
    std::mt19937 random(12345);
    for (const int t : {1, 3, 8, 16, 255}) {
        const BCH bch(t);
        const auto original = random_data(random);
        const auto helper = bch.compute_syndrome(original);
        const auto packed = bch.pack_syndromes(helper);
        CHECK(bch.unpack_syndromes(packed) == helper);
        CHECK(packed.size() == (static_cast<std::size_t>(20 * t) + 7) / 8);
        auto noisy = original;
        flip(noisy, 511);
        CHECK(bch.recover(noisy, bch.unpack_syndromes(packed)).data == original);
    }
    const BCH bch;
    CHECK(bch.syndrome_storage_bits() == 160);
    CHECK(bch.syndrome_storage_bytes() == 20);
    BCH::Syndromes all_max(17, 1023);
    all_max[0] = 0;
    CHECK(bch.pack_syndromes(all_max) == BCH::Bytes(20, 0xFF));
    CHECK(bch.unpack_syndromes(BCH::Bytes(20, 0xFF)) == all_max);

    const BCH one(1);
    // 10-bit fields 1 and 1023: 0000000001 1111111111 0000.
    CHECK(one.pack_syndromes({0, 1, 1023}) == BCH::Bytes({0x00, 0x7F, 0xF0}));
    CHECK(one.unpack_syndromes({0x00, 0x7F, 0xF0}) == BCH::Syndromes({0, 1, 1023}));
    invalid([&] { one.unpack_syndromes({0x00, 0x7F, 0xF1}); });
}

void test_invalid_inputs() {
    invalid([] { BCH invalid_bch(0); });
    invalid([] { BCH invalid_bch(-1); });
    invalid([] { BCH invalid_bch(256); });
    const BCH bch;
    const BCH::Bytes data(BCH::input_bytes, 0);
    const auto helper = bch.compute_syndrome(data);
    for (const std::size_t length : {0U, 32U, 63U, 65U, 512U}) {
        invalid([&] { bch.compute_syndrome(BCH::Bytes(length)); });
        invalid([&] { bch.recover(BCH::Bytes(length), helper); });
        invalid([&] { BCH::to_bitstring(BCH::Bytes(length)); });
    }
    for (const std::size_t length : {0U, 16U, 18U}) {
        invalid([&] { bch.recover(data, BCH::Syndromes(length, 0)); });
        invalid([&] { bch.pack_syndromes(BCH::Syndromes(length, 0)); });
    }
    for (const int bad : {-1, 1024}) {
        auto broken = helper;
        broken[1] = bad;
        invalid([&] { bch.recover(data, broken); });
        invalid([&] { bch.pack_syndromes(broken); });
    }
    auto bad_dummy = helper;
    bad_dummy[0] = 1;
    invalid([&] { bch.recover(data, bad_dummy); });
    invalid([&] { bch.unpack_syndromes(BCH::Bytes(19)); });
    invalid([&] { bch.unpack_syndromes(BCH::Bytes(21)); });
    invalid([] { BCH::from_bitstring(std::string(511, '0')); });
    invalid([] { BCH::from_bitstring(std::string(513, '0')); });
    invalid([] { BCH::from_bitstring(std::string(512, '2')); });
}

void test_uncorrectable_and_shortening() {
    const BCH bch;
    const BCH::Bytes zero(BCH::input_bytes, 0);
    // Valid mother-code syndromes for a bit outside the retained 512 positions.
    auto outside = bch.compute_syndrome(zero);
    for (int j = 1; j <= 16; ++j) {
        outside[j] = polynomial_power(2, (j * 700) % 1023);
    }
    const auto rejected = bch.recover(zero, outside);
    CHECK(rejected.status == BCH::DecodeStatus::Uncorrectable);
    CHECK(rejected.data.empty());
    CHECK(rejected.corrected_positions.empty());

    std::mt19937 random(0xBAD512U);
    int failures = 0;
    for (const int errors : {9, 10, 16, 32, 64, 128}) {
        for (int trial = 0; trial < 12; ++trial) {
            const auto enrolled = random_data(random);
            const auto helper = bch.compute_syndrome(enrolled);
            auto noisy = enrolled;
            for (const auto position : error_positions(random, errors)) {
                flip(noisy, position);
            }
            const auto result = bch.recover(noisy, helper);
            if (!result.decoded()) {
                ++failures;
                CHECK(result.data.empty());
                CHECK(result.corrected_positions.empty());
            } else {
                // Beyond t, a different valid candidate is possible. Do not make
                // the mathematically incorrect assertion that all >t errors fail.
                CHECK(result.corrected_positions.size() <= 8);
                CHECK(bch.compute_syndrome(result.data) == helper);
                CHECK(result.data != enrolled);
            }
        }
    }
    CHECK(failures > 0);
}

void test_independent_codeword_and_detection_limit() {
    // Build the binary generator from cyclotomic roots using independent
    // polynomial arithmetic, rather than the production decoder's log tables.
    std::set<int> roots;
    for (int j = 1; j <= 16; ++j) {
        int exponent = j;
        do {
            roots.insert(exponent);
            exponent = (2 * exponent) % 1023;
        } while (exponent != j);
    }
    std::vector<int> generator{1};
    for (const int exponent : roots) {
        const int root = polynomial_power(2, exponent);
        std::vector<int> next(generator.size() + 1, 0);
        for (std::size_t i = 0; i < generator.size(); ++i) {
            next[i] ^= polynomial_multiply(generator[i], root);
            next[i + 1] ^= generator[i];
        }
        generator = std::move(next);
    }
    CHECK(generator.size() == 81); // Degree 80 for this t=8 mother code.
    BCH::Bytes codeword(BCH::input_bytes, 0);
    int weight = 0;
    for (std::size_t i = 0; i < generator.size(); ++i) {
        CHECK(generator[i] == 0 || generator[i] == 1);
        if (generator[i] == 1) {
            flip(codeword, i);
            ++weight;
        }
    }
    CHECK(weight >= 17);
    const BCH bch;
    const BCH::Bytes enrolled(BCH::input_bytes, 0);
    const auto helper = bch.compute_syndrome(enrolled);
    CHECK(reference_syndromes(codeword, 8) == helper);
    CHECK(bch.compute_syndrome(codeword) == helper);
    expect_recovery(bch, codeword, {0, 255, 256, 511});

    const auto collision = bch.recover(codeword, helper);
    CHECK(collision.status == BCH::DecodeStatus::NoChange);
    CHECK(collision.data != enrolled);
    auto noisy = codeword;
    flip(noisy, 511);
    const auto different_candidate = bch.recover(noisy, helper);
    CHECK(different_candidate.status == BCH::DecodeStatus::Corrected);
    CHECK(different_candidate.data == codeword);
    CHECK(different_candidate.data != enrolled);
}
} // namespace

int main() {
    try {
        test_all_positions_and_random_errors();
        test_independent_syndromes_and_bit_packing();
        test_helper_serialization();
        test_invalid_inputs();
        test_uncorrectable_and_shortening();
        test_independent_codeword_and_detection_limit();
        std::cout << "PASS: 6 test groups, " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
