#ifndef MULTIMODAL_BIOMETRIC_BCH_HPP
#define MULTIMODAL_BIOMETRIC_BCH_HPP

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

// Syndrome-based secure-sketch component, adapted from DID-Wallet's BCH API.
// The input is 512 already-binarized bits packed into exactly 64 bytes.
// This class does not generate keys, authenticate helper data, or binarize floats.
class BCH {
public:
    using Bytes = std::vector<unsigned char>;
    using Syndromes = std::vector<int>;

    static constexpr int field_degree = 10;
    static constexpr int mother_code_bits = (1 << field_degree) - 1;
    static constexpr int primitive_polynomial = 0x409; // x^10 + x^3 + 1
    static constexpr std::size_t input_bits = 512;
    static constexpr std::size_t input_bytes = input_bits / 8;
    static constexpr int max_error_capacity = 255;

    enum class DecodeStatus { NoChange, Corrected, Uncorrectable };

    struct RecoveryResult {
        DecodeStatus status = DecodeStatus::Uncorrectable;
        Bytes data; // Empty on Uncorrectable; never an unchecked partial result.
        std::vector<std::size_t> corrected_positions; // Zero-based, MSB first.

        // A syndrome-consistent candidate within t flips was found.
        // This is NOT a claim of key equality/authenticity beyond the t-error bound.
        bool decoded() const noexcept { return status != DecodeStatus::Uncorrectable; }
    };

    explicit BCH(int t = 8);
    int error_capacity() const noexcept { return t_; }
    std::size_t syndrome_storage_bits() const noexcept;
    std::size_t syndrome_storage_bytes() const noexcept;

    // Keeps the original indexing convention: s[0] == 0, s[1..2t] are GF elements.
    Syndromes compute_syndrome(const Bytes& data) const;
    RecoveryResult recover(const Bytes& noisy_data, const Syndromes& saved_syndromes) const;

    // Compact, fixed-width 10-bit fields, MSB first; omits the unused s[0].
    // There is no embedded metadata: retain format version and t separately.
    Bytes pack_syndromes(const Syndromes& syndromes) const;
    Syndromes unpack_syndromes(const Bytes& packed) const;

    static Bytes from_bitstring(std::string_view bits);
    static std::string to_bitstring(const Bytes& data);

private:
    int t_;
    std::vector<int> alpha_to_;
    std::vector<int> index_of_;

    void init_galois();
    int gf_mul(int a, int b) const;
    int gf_inv(int a) const;
    static void validate_data(const Bytes& data);
    void validate_syndromes(const Syndromes& syndromes) const;
};

#endif
