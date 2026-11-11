// Deliberately deterministic replay stream for tests ONLY. Not a CSPRNG.
// This library is excluded from normal builds and never linked into FL targets.
#include "utils/prng/prng.h"
class ReplayPrng : public PRNG {
 public:
    result_type operator()() override {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return static_cast<result_type>(state >> 32);
    }
 private:
    std::uint64_t state = 0x123456789abcdefULL;
};
extern "C" PRNG* createEngineInstance() { return new ReplayPrng(); }
