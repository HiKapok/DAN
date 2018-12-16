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
#include "common.h"

//__attribute__((always_inline))
// template<typename T, typename std::is_same<float, typename std::remove_cv<T>::type>::value>
void atomic_float_add(volatile float* ptr, const float operand)
{
    assert(is_aligned(ptr, 4));

    volatile int32_t* iptr = reinterpret_cast<volatile int32_t*>(ptr);
    int32_t expected = *iptr;

    while (true)
    {
        const float value = binary_cast<float>(expected);
        const int32_t new_value = binary_cast<int32_t>(value + operand);
        const int32_t actual = __sync_val_compare_and_swap(iptr, expected, new_value);
        if (actual == expected)
            return;
        expected = actual;
    }
}


// void atomic_store_int32(int32_t* p_ai, int32_t val)
// {
//     int32_t ai_was;

//     ai_was = *p_ai;
//     do { ai_was = __sync_val_compare_and_swap(p_ai, ai_was, val) ;
// }

// void atomic_store_float(float* ptr, float val)
// {
//     assert(is_aligned(ptr, 4));

//     volatile int32_t* iptr = reinterpret_cast<volatile int32_t*>(ptr);
//     int32_t expected = *iptr;

//     while (true)
//     {
//         const int32_t new_value = binary_cast<int32_t>(val);
//         const int32_t actual = __sync_val_compare_and_swap(iptr, expected, new_value);
//         if (actual == expected)
//             return;
//         expected = actual;
//     }
// }
