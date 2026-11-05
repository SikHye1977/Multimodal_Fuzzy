// Batch evaluation adapter only. This does not implement FE.Gen/FE.Rep or keys.
#include "BCH.hpp"
#include <algorithm>
#include <cstddef>

extern "C" int fusion_bch_recover(int t, const unsigned char* enrolled,
                                  const unsigned char* query, std::size_t n,
                                  unsigned char* outcomes) noexcept {
    try {
        BCH bch(t);
        for (std::size_t i = 0; i < n; ++i) {
            BCH::Bytes e(enrolled + 64*i, enrolled + 64*(i+1));
            BCH::Bytes q(query + 64*i, query + 64*(i+1));
            const auto helper = bch.compute_syndrome(e);
            const auto result = bch.recover(q, helper);
            // 0: no candidate; 1: exact enrolled bits; 2: wrong consistent candidate.
            outcomes[i] = !result.decoded() ? 0 : (result.data == e ? 1 : 2);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
