#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "../../include/metric/metric.h"

inline int FastMod(const int input, const int ceil)
{
    int result = input % ceil;
    return result >= 0 ? result : result + ceil;
}

inline uint64_t EXPmodule(uint32_t base, uint16_t exp, uint32_t mod)
{
    uint64_t res = 1;

    while (exp > 0)
    {
        if (exp & 1)
        {
            res = FastMod(res * base, mod);
        }
        exp >>= 1;
        base = FastMod(base * base, mod);
    }
    return res;
}

inline int modulo(int i, int n)
{
    return ((i % n) + n) % n;
}

#endif 