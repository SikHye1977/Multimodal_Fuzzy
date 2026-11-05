#include "BCH.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

BCH::BCH(int t) : t_(t) {
    if (t < 1 || t > max_error_capacity) {
        throw std::invalid_argument("BCH error capacity t must be in [1, 255]");
    }
    init_galois();
}

void BCH::init_galois() {
    alpha_to_.assign(mother_code_bits + 1, 0);
    index_of_.assign(mother_code_bits + 1, -1);
    int value = 1;
    for (int exponent = 0; exponent < mother_code_bits; ++exponent) {
        if (value <= 0 || value > mother_code_bits || index_of_[value] != -1) {
            throw std::logic_error("BCH polynomial does not generate the full field");
        }
        alpha_to_[exponent] = value;
        index_of_[value] = exponent;
        value <<= 1;
        if ((value & (1 << field_degree)) != 0) {
            value ^= primitive_polynomial;
        }
    }
    if (value != 1) {
        throw std::logic_error("BCH field period is not 1023");
    }
    alpha_to_[mother_code_bits] = 1;
}

int BCH::gf_mul(int a, int b) const {
    if (a == 0 || b == 0) {
        return 0;
    }
    return alpha_to_[(index_of_[a] + index_of_[b]) % mother_code_bits];
}

int BCH::gf_inv(int a) const {
    if (a == 0) {
        throw std::logic_error("Cannot invert zero in GF(2^10)");
    }
    return alpha_to_[mother_code_bits - index_of_[a]];
}

void BCH::validate_data(const Bytes& data) {
    if (data.size() != input_bytes) {
        throw std::invalid_argument("BCH input must contain exactly 64 packed bytes (512 bits)");
    }
}

void BCH::validate_syndromes(const Syndromes& syndromes) const {
    if (syndromes.size() != static_cast<std::size_t>(2 * t_ + 1) || syndromes[0] != 0) {
        throw std::invalid_argument("BCH syndromes must have 2*t+1 elements with s[0] == 0");
    }
    for (const int value : syndromes) {
        if (value < 0 || value > mother_code_bits) {
            throw std::invalid_argument("BCH syndrome elements must be in [0, 1023]");
        }
    }
}

BCH::Syndromes BCH::compute_syndrome(const Bytes& data) const {
    validate_data(data);
    Syndromes result(static_cast<std::size_t>(2 * t_ + 1), 0);
    // Bit i is the coefficient of x^i; positions 512..1022 are fixed at zero.
    for (int j = 1; j <= 2 * t_; ++j) {
        int value = 0;
        for (std::size_t i = 0; i < input_bits; ++i) {
            if (((data[i / 8] >> (7 - i % 8)) & 1U) != 0) {
                const int exponent = (j * static_cast<int>(i)) % mother_code_bits;
                value ^= alpha_to_[exponent];
            }
        }
        result[static_cast<std::size_t>(j)] = value;
    }
    return result;
}

BCH::RecoveryResult BCH::recover(const Bytes& noisy_data, const Syndromes& saved_syndromes) const {
    validate_data(noisy_data);
    validate_syndromes(saved_syndromes);
    Syndromes errors = compute_syndrome(noisy_data);
    bool has_error = false;
    for (int j = 1; j <= 2 * t_; ++j) {
        errors[j] ^= saved_syndromes[j];
        has_error = has_error || errors[j] != 0;
    }
    if (!has_error) {
        return {DecodeStatus::NoChange, noisy_data, {}};
    }

    // Berlekamp-Massey. Allocate through 2t, not t: out-of-radius inputs can
    // produce a locator degree above t and must not cause an out-of-bounds read.
    const int count = 2 * t_;
    std::vector<int> locator(static_cast<std::size_t>(count + 1), 0);
    std::vector<int> previous(locator.size(), 0);
    locator[0] = previous[0] = 1;
    int degree = 0;
    int shift = 1;
    int last_discrepancy = 1;

    for (int step = 0; step < count; ++step) {
        int discrepancy = errors[step + 1];
        for (int i = 1; i <= degree; ++i) {
            discrepancy ^= gf_mul(locator[i], errors[step + 1 - i]);
        }
        if (discrepancy == 0) {
            ++shift;
            continue;
        }
        const auto old_locator = locator;
        const int factor = gf_mul(discrepancy, gf_inv(last_discrepancy));
        for (int i = 0; i + shift <= count; ++i) {
            locator[i + shift] ^= gf_mul(factor, previous[i]);
        }
        if (2 * degree <= step) {
            degree = step + 1 - degree;
            previous = old_locator;
            last_discrepancy = discrepancy;
            shift = 1;
        } else {
            ++shift;
        }
    }
    if (degree < 1 || degree > t_) {
        return {};
    }

    // Chien search over the entire mother code. A root in a shortened-away
    // position is invalid; ignoring that root could otherwise accept a bad result.
    std::vector<std::size_t> positions;
    for (int position = 0; position < mother_code_bits; ++position) {
        const int x = alpha_to_[(mother_code_bits - position) % mother_code_bits];
        int value = locator[degree];
        for (int j = degree - 1; j >= 0; --j) {
            value = gf_mul(value, x) ^ locator[j];
        }
        if (value == 0) {
            if (position >= static_cast<int>(input_bits)) {
                return {};
            }
            positions.push_back(static_cast<std::size_t>(position));
        }
    }
    if (positions.size() != static_cast<std::size_t>(degree)) {
        return {};
    }

    Bytes recovered = noisy_data;
    for (const auto position : positions) {
        recovered[position / 8] ^= static_cast<unsigned char>(1U << (7 - position % 8));
    }
    if (compute_syndrome(recovered) != saved_syndromes) {
        return {};
    }
    return {DecodeStatus::Corrected, std::move(recovered), std::move(positions)};
}

std::size_t BCH::syndrome_storage_bits() const noexcept {
    return static_cast<std::size_t>(2 * t_ * field_degree);
}

std::size_t BCH::syndrome_storage_bytes() const noexcept {
    return (syndrome_storage_bits() + 7) / 8;
}

BCH::Bytes BCH::pack_syndromes(const Syndromes& syndromes) const {
    validate_syndromes(syndromes);
    Bytes packed(syndrome_storage_bytes(), 0);
    std::size_t cursor = 0;
    for (std::size_t j = 1; j < syndromes.size(); ++j) {
        for (int bit = field_degree - 1; bit >= 0; --bit, ++cursor) {
            if (((syndromes[j] >> bit) & 1) != 0) {
                packed[cursor / 8] |= static_cast<unsigned char>(1U << (7 - cursor % 8));
            }
        }
    }
    return packed;
}

BCH::Syndromes BCH::unpack_syndromes(const Bytes& packed) const {
    if (packed.size() != syndrome_storage_bytes()) {
        throw std::invalid_argument("Packed BCH syndrome length does not match t");
    }
    const auto remainder = syndrome_storage_bits() % 8;
    if (remainder != 0 && (packed.back() & ((1U << (8 - remainder)) - 1U)) != 0) {
        throw std::invalid_argument("Packed BCH syndrome padding must be zero");
    }
    Syndromes result(static_cast<std::size_t>(2 * t_ + 1), 0);
    std::size_t cursor = 0;
    for (std::size_t j = 1; j < result.size(); ++j) {
        for (int bit = 0; bit < field_degree; ++bit, ++cursor) {
            result[j] = (result[j] << 1) | ((packed[cursor / 8] >> (7 - cursor % 8)) & 1);
        }
    }
    return result;
}

BCH::Bytes BCH::from_bitstring(std::string_view bits) {
    if (bits.size() != input_bits) {
        throw std::invalid_argument("BCH bitstring must contain exactly 512 binary characters");
    }
    Bytes data(input_bytes, 0);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] != '0' && bits[i] != '1') {
            throw std::invalid_argument("BCH bitstring may contain only '0' and '1'");
        }
        if (bits[i] == '1') {
            data[i / 8] |= static_cast<unsigned char>(1U << (7 - i % 8));
        }
    }
    return data;
}

std::string BCH::to_bitstring(const Bytes& data) {
    validate_data(data);
    std::string bits(input_bits, '0');
    for (std::size_t i = 0; i < input_bits; ++i) {
        if (((data[i / 8] >> (7 - i % 8)) & 1U) != 0) {
            bits[i] = '1';
        }
    }
    return bits;
}
