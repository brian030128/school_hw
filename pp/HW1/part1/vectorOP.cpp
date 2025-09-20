#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

#define CLAMP_VALUE 9.999999f
void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_int ones = _pp_vset_int(1);
  __pp_vec_int zeroes = _pp_vset_int(0);
  __pp_vec_float clamp = _pp_vset_float(CLAMP_VALUE);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int remaining = N-i;
    __pp_mask mask;
    if(remaining >= VECTOR_WIDTH){
      mask = _pp_init_ones();
    } else{
      mask = _pp_init_ones(remaining);
    }

    __pp_vec_float vecValues;
    __pp_vec_int vecExps;

    _pp_vload_float(vecValues, &values[i], mask);
    _pp_vload_int(vecExps, &exponents[i], mask);

    __pp_vec_float vecResult = _pp_vset_float(1.0f);
    __pp_mask activeMask = mask;
    _pp_vgt_int(activeMask, vecExps, zeroes, activeMask);
    while(_pp_cntbits(activeMask) > 0) {
      _pp_vmult_float(vecResult, vecResult, vecValues, activeMask);
      _pp_vsub_int(vecExps, vecExps, ones, activeMask);
      _pp_vgt_int(activeMask, vecExps, zeroes, activeMask);
      
    }

    __pp_mask clampMask;
     _pp_vgt_float(clampMask, vecResult, clamp,mask);
     _pp_vset_float(vecResult, CLAMP_VALUE, clampMask);

    _pp_vstore_float(&output[i], vecResult, mask);
  }

}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}