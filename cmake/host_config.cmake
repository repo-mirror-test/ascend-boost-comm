set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)

string(REGEX REPLACE "[^A-Za-z0-9_]" "" NAMESPACE "${NAMESPACE}")

message(STATUS "The namespace for infra would be: ^${NAMESPACE}$")

if(NOT DEFINED NO_WERROR)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>")
endif()

add_compile_options(
    -fexceptions
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++17;-pipe;-Wno-unused-parameter;-Wno-ignored-qualifiers>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wformat=0;-Wno-strict-overflow;-fno-strict-aliasing>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fPIC;-fstack-protector-all;-Wl,--build-id=none>"
    "$<$<NOT:$<CONFIG:Release>>:-g>"
)

if(NOT NAMESPACE STREQUAL "")
    add_compile_definitions(OpSpace=${NAMESPACE}) # TODO: 第一次构建，需要指定NAMESPACE，存储在cmake的CACHE变量中
endif()

add_compile_definitions(
    "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=$<BOOL:${USE_CXX11_ABI}>>"
    "$<$<NOT:$<CONFIG:Release>>:_DEBUG>"
)

set(LD_FLAGS_GLOBAL "-shared;-rdynamic;-ldl;-Wl,-z,relro;-Wl,-z,now")
set(LD_FLAGS_GLOBAL "${LD_FLAGS_GLOBAL};-Wl,-z,noexecstack;-Wl,--build-id=none")

add_link_options(
    "$<$<CONFIG:Release>:-s>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${LD_FLAGS_GLOBAL};-fexceptions>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${LD_FLAGS_GLOBAL};-pie;-fPIE>"
)

if(NPU_FORMAT_H_VALID)
    message(STATUS "NPUFormat.h is available")
    add_compile_definitions(NPU_FORMAT_H_VALID)
endif()

if(BUILD_TEST_FRAMEWORK OR USE_UNIT_TEST OR USE_PYTHON_TEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O3")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()
