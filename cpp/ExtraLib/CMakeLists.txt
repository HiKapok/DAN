CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(extra_lib)

find_package(CUDA REQUIRED)

EXECUTE_PROCESS(COMMAND python3 -c "import os; print(os.getcwd(), end='', flush=True)" OUTPUT_VARIABLE CWD)
MESSAGE(STATUS "Found CWD: " ${CWD})

EXECUTE_PROCESS(COMMAND python3 -c "import subprocess; process = subprocess.Popen('nvidia-smi -i 0 --query-gpu=name --format=csv'.split(), stdout=subprocess.PIPE); output, _ = process.communicate(); output = str(output); device_capability_map = {
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla C2075' : '20',
    'Tesla C2050' : '20',
    'Tesla C2070' : '20',
    'Tesla V100'  : '70',
    'Tesla P100'  : '60',
    'Tesla P40'   : '61',
    'Tesla P4'    : '61',
    'Tesla M60'   : '52',
    'Tesla M40'   : '52',
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla K10'   : '30',
    'GeForce GTX 1080 Ti' : '61'
}; cap = '61';
for k, v in device_capability_map.items():
    if k in output:
        cap = v
        break
print('gencode arch=compute_' + cap + ',code=sm_' + cap)" OUTPUT_VARIABLE GPU_CAPABILITY)
MESSAGE(STATUS "Found GPU_CAPABILITY: " ${GPU_CAPABILITY})

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --keep --keep-dir ${CWD} -${GPU_CAPABILITY} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --keep --keep-dir ${CWD} -gencode arch=compute_61,code=sm_61 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA=1")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)

EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)


MESSAGE(STATUS "Found TF_INC: " ${TF_INC})
MESSAGE(STATUS "Found TF_INC_EXTERNAL: " ${TF_INC}/external/nsync/public)
MESSAGE(STATUS "Found TF_LIB: " ${TF_LIB})


INCLUDE_DIRECTORIES(${TF_INC})
INCLUDE_DIRECTORIES(${TF_INC}/external/nsync/public)
LINK_DIRECTORIES(${TF_LIB})

# approach 2

ADD_LIBRARY(extra_lib SHARED
  small_mining_match.h
  small_mining_match.cc
  dynamic_anchor_routing.cc
  dynamic_anchor_routing.h
  common.h
  common.cc
  )

TARGET_LINK_LIBRARIES(extra_lib tensorflow_framework ${CUDA_LIBRARIES})
