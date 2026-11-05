#include "BCH.hpp"

#include <iostream>

int main() {
    const BCH bch(8);
    const BCH::Bytes enrolled(BCH::input_bytes, 0xA5); // Synthetic example only.
    const auto helper = bch.pack_syndromes(bch.compute_syndrome(enrolled));

    auto query = enrolled;
    for (const std::size_t position : {0U, 255U, 256U, 511U}) {
        query[position / 8] ^= static_cast<unsigned char>(1U << (7 - position % 8));
    }
    const auto result = bch.recover(query, bch.unpack_syndromes(helper));
    if (!result.decoded()) {
        std::cerr << "No syndrome-consistent candidate within the correction radius.\n";
        return 1;
    }
    // Exact equality can be checked here because this is a synthetic test.
    // A real FE must perform its separate key verification after reconstruction.
    const bool exact_match = result.data == enrolled;
    std::cout << "Input: " << BCH::input_bits << " bits\n"
              << "Correction capacity: " << bch.error_capacity() << " bits\n"
              << "Packed syndrome: " << helper.size() << " bytes\n"
              << "Corrected bits: " << result.corrected_positions.size() << '\n'
              << "Exact reconstruction: " << (exact_match ? "PASS" : "FAIL") << '\n';
    return exact_match ? 0 : 1;
}
