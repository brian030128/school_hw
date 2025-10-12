#include "SIMDInstructionSet.h"
#include <iostream>
#include "Xoshiro256Plus.h"


int main(int argc, char* argv[]) {
    SEFUtility::RNG::Xoshiro256PlusSIMD8 rng(123);
    auto test = rng.next8();
    std::cout << "Test random number: " << test[0] << ", " << test[1] << ", " << test[2] << ", " << test[3] << ", "
         << test[4] << ", " << test[5] << ", " << test[6] << ", " << test[7] << std::endl;
}