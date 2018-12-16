// Copyright (c) 2018 Changan Wang

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef COMMON_H_
#define COMMON_H_

#include <cstdlib>
#include <cassert>
#include <cstdint>

// atomic addition for float using gcc built-in functions for atomic memory access
// this code snippet borrowed from https://codereview.stackexchange.com/questions/135852/atomic-floating-point-addition
template <typename Target, typename Source>
__attribute__((always_inline)) Target binary_cast(Source s)
{
    static_assert(sizeof(Target) == sizeof(Source), "binary_cast: 'Target' must has the same size as 'Source'");
    union
    {
        Source  m_source;
        Target  m_target;
    } u;

    u.m_source = s;
    return u.m_target;
}

template <typename T>
__attribute__((always_inline)) bool is_pow2(const T x)
{
    return (x & (x - 1)) == 0;
}

template <typename T>
__attribute__((always_inline)) bool is_aligned(const T ptr, const size_t alignment)
{
    assert(alignment > 0);
    assert(is_pow2(alignment));

    const uintptr_t p = (uintptr_t)ptr;
    return (p & (alignment - 1)) == 0;
}

extern void atomic_float_add(volatile float* ptr, const float operand);
// extern void atomic_store_int32(int32_t* p_ai, int32_t val);
// extern void atomic_store_float(float* ptr, float val);

#endif // COMMON_H_
