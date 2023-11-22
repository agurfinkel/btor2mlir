#include "cex.h"
#include <iostream>

extern "C" {

void __VERIFIER_error() {
  std::cout << "[sea] __VERIFIER_error was executed" << std::endl;
  exit(1);
}

void __VERIFIER_assert(bool x, int property) {
  std::cout << "[sea] __VERIFIER_assert was called for property: " << property << std::endl;
}

void btor2mlir_print_input_num(unsigned num, unsigned value, unsigned width) { 
  std::cout << "input, " << num << ", " << value << ", " << width << std::endl;
}

void btor2mlir_print_state_num(unsigned num, unsigned value, unsigned width) { 
  std::cout << "state, " << num << ", " << value << ", " << width << std::endl;
}

bool __seahorn_get_value_i1(int ctr, bool *g_arr, int g_arr_sz) {
  std::cout << "[sea] __seahorn_get_value_i1(" << ctr << ", " << g_arr_sz << ")\n";
  if (ctr >= g_arr_sz) {
    std::cout << "\tout-of-bounds index\n";
    return 0;
  } else {
    return g_arr[ctr];
  }
}

#define get_value_helper(ctype, llvmtype)                               \
  ctype __seahorn_get_value_ ## llvmtype (int ctr, ctype *g_arr, int g_arr_sz) { \
    std::cout << "[sea] __seahorn_get_value_(" << #llvmtype ", " << ctr << ", " << g_arr_sz << ")\n"; \
    if (ctr >= g_arr_sz) {						\
      std::cout << "\tout-of-bounds index\n";				\
      return 0;								\
    }									\
    else { 								\
      return g_arr[ctr];						\
    }									\
  }

#define get_value_int(bits) get_value_helper(int ## bits ## _t, i ## bits)

get_value_int(64);
get_value_int(32);
get_value_int(16);
get_value_int(8);

get_value_helper(intptr_t, ptr_internal);

}
