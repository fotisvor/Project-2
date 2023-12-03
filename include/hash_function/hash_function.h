#ifndef HASH_FUNCTION_H
#define HASH_FUNCTION_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "../math_utils/mathutils.h"

template <typename T> class HashFunction
{
 private:
    const uint16_t k,d,m,M;
    const double w;
    std::vector<double> Stransform;
    std::vector<double> a;
 public:
    HashFunction(const uint16_t k, const uint16_t d, const uint32_t m, const uint32_t M, const double w)
        : k(k), d(d), m(m), M(M), w(w), Stransform(d), a(d)
    {
        unsigned Seeding = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(Seeding);
        std::uniform_real_distribution<double> Distribute(0.0, w);

        for (size_t i=0;i!= d; i++)
        {
            Stransform[i] = Distribute(generator);
        }
    }

    ~HashFunction() = default;

    uint32_t HashFunctionConstructor(const std::vector<T> &pixels)
    {
        uint32_t HashValue = 0;

        for (size_t i = 0; i != d; i++)
        {
            a[i] = floor((pixels[i] - Stransform[i]) / w);
        }

        std::reverse(a.begin(), a.end());

        for (size_t i = 0; i != d; i++)
        {
            HashValue += (modulo(a[i], M) * EXPmodule(m, i, M)) % M;
        }

        return FastMod(HashValue, M);
    }
};

template <typename T> class Amplified_HASH_function
{

  private:
    const uint16_t k,d;
    const uint32_t m,M;
    const double w;

    std::vector<HashFunction<T>> HI;

  public:
    Amplified_HASH_function(const uint16_t k, const uint16_t d, const uint32_t m, const uint32_t M, const double w)
        : k(k), d(d), m(m), M(M), w(w)
    {

        for (size_t i = 0; i != k; i++)
        {
            HI.push_back(HashFunction<T>(k, d, m, M, w));
        }
    }

    ~Amplified_HASH_function() = default;

    size_t amplfied_FunctionConstructor(const std::vector<T> &pixels)
    {

        std::string res = "";
        char *p_end = nullptr;

        for (size_t i = 0; i != k; i++)
        {
            res += std::to_string(HI[i].HashFunctionConstructor(pixels));
        }

        return std::strtoull(res.c_str(), &p_end, 10);
    }
};

#endif 
