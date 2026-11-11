set(FEDERATED1_TEST_SOURCES
    federated1/federated-encrypted.cpp
    federated1/federated-reference.cpp
    federated1/collective-conversion-keygen.cpp
    federated1/crypto-execution-test.cpp
    federated1/coefficient-codec-test.cpp
    federated1/ring-conversion-test.cpp
    federated1/threshold-ckks-test.cpp
    federated1/collective-conversion-keygen-test.cpp
    federated1/federated-simulation-test.cpp
    federated1/protocol-stage1-test.cpp
    federated1/protocol-setup-test.cpp
    federated1/protocol-stage3-test.cpp
    federated1/protocol-stage4-test.cpp
    federated1/protocol-stage5-test.cpp
    federated1/protocol-v17-law-test.cpp
    federated1/protocol-contract-test.cpp
    federated1/external-input-crypto-engine-test.cpp
)

set(FEDERATED1_PROTOCOL_SOURCES
    federated1/crypto-execution.cpp
    federated1/protocol-randomness.cpp
    federated1/coefficient-codec.cpp
    federated1/ring-conversion.cpp
    federated1/threshold-ckks.cpp

    federated1/sparsification.cpp
    federated1/chunk-packing.cpp
    federated1/protocol-parameters.cpp
    federated1/deterministic-randomness.cpp
    federated1/rns-polynomial.cpp
    federated1/coefficient-compression.cpp
    federated1/protocol-v17-decryption.cpp
    federated1/protocol-setup.cpp
    federated1/survivor-keys.cpp
    federated1/seeded-rlwe.cpp
    federated1/client-to-collective-kswitch.cpp
    federated1/protocol-sparse-round.cpp
    federated1/protocol-ring-conversion.cpp
    federated1/protocol-sparse-path.cpp
    federated1/seeded-dense-rlwe.cpp
    federated1/direct-rlwe-aggregation.cpp
    federated1/protocol-full-model-path.cpp
    federated1/protocol-auxiliary-path.cpp
    federated1/protocol-crypto-context.cpp
    federated1/protocol-pre-smudging-noise.cpp
    federated1/protocol-contract.cpp
    federated1/support-bitmap.cpp
    federated1/external-input-crypto-engine.cpp
)

function(configure_federated1_target target_name)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/federated1
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/pke/include
        ${CMAKE_SOURCE_DIR}/src/binfhe/include
        ${CMAKE_BINARY_DIR}/src/core
    )

    if(BUILD_SHARED)
        target_link_libraries(${target_name} PRIVATE
            OPENFHEpke
            OPENFHEcore
        )
    else()
        target_link_libraries(${target_name} PRIVATE
            OPENFHEpke_static
            OPENFHEcore_static
        )
    endif()
endfunction()

add_library(federated1_protocol STATIC
    ${FEDERATED1_PROTOCOL_SOURCES}
)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
target_include_directories(federated1_protocol PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(federated1_protocol PUBLIC Threads::Threads)
set_target_properties(federated1_protocol PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
configure_federated1_target(federated1_protocol)

add_executable(federated1 EXCLUDE_FROM_ALL
    federated1/main.cpp
    ${FEDERATED1_TEST_SOURCES}
)
configure_federated1_target(federated1)
target_link_libraries(federated1 PRIVATE federated1_protocol)

add_executable(codec_benchmark EXCLUDE_FROM_ALL federated1/codec-benchmark.cpp)
configure_federated1_target(codec_benchmark)
target_link_libraries(codec_benchmark PRIVATE federated1_protocol)

add_executable(gaussian_benchmark EXCLUDE_FROM_ALL federated1/gaussian-benchmark.cpp)
configure_federated1_target(gaussian_benchmark)
target_link_libraries(gaussian_benchmark PRIVATE federated1_protocol)
add_library(gaussian_test_prng SHARED EXCLUDE_FROM_ALL federated1/gaussian-test-prng.cpp)
target_include_directories(gaussian_test_prng PRIVATE ${CMAKE_SOURCE_DIR}/src/core/include)

if(BUILD_FEDERATED_PYTHON_BINDINGS)
    # Development.Module was introduced in CMake 3.18.  OpenFHE still
    # supports CMake 3.16, where Development also defines Python3::Module.
    if(CMAKE_VERSION VERSION_LESS 3.18)
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    else()
        find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    endif()
    # NumPy 2 array layouts require pybind11 2.12 or newer.
    find_package(pybind11 2.12 CONFIG REQUIRED)

    pybind11_add_module(ciphertext_fl_cpp MODULE
        federated1/python-bindings.cpp
    )
    configure_federated1_target(ciphertext_fl_cpp)
    target_link_libraries(ciphertext_fl_cpp PRIVATE federated1_protocol)
    target_compile_definitions(ciphertext_fl_cpp PRIVATE
        FEDERATED1_PYBIND_MODULE=1
    )
    set_target_properties(ciphertext_fl_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
endif()
