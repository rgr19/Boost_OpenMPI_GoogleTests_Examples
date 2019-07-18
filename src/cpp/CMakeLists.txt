project(Cpp_Example LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.13)

#set ( CMAKE_BUILD_TYPE RELEASE )

message("Build EXE: ${PROJECT_NAME}")

set(BOOST_LIBS
        python27
        numpy27
        )

set(SOURCES
        main.std.stl.NN.layers.vararg.cpp
        NNet.hpp NNet.hxx
        DGen.hxx DGen.hpp
        MUtil.h
        CEng.hxx CEng.hpp tests.hpp
        )

if (DO_BOOST)
    add_executable_boost(${PROJECT_NAME}_BOOST "${SOURCES}" "${BOOST_LIBS}")
else ()
    add_executable(${PROJECT_NAME}_BOOST ${SOURCES})
    config_bin_output(${PROJECT_NAME}_BOOST)
endif ()

set_target_properties(${PROJECT_NAME}_BOOST PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

if (DO_PYTHON)
    link_python(${PROJECT_NAME}_BOOST)
endif ()

link_resources(${PROJECT_NAME}_BOOST)

#project(cpp_example_1 languages c cxx)
#add_executable(${project_name} ${sources})
#set_target_properties(${project_name} properties cxx_standard 17 cxx_standard_required yes cxx_extensions on)

ppr_blue("Build EXE: ${PROJECT_NAME}")
add_executable(${PROJECT_NAME} MUtil.h matrix_operations.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

if (0)

    # create simple lib
    ppr_blue("Build LIBRARY: simple_cpp_lib...")

    add_library(${PROJECT_NAME}_Lib simple_lib.cpp simple_lib.hpp)
    config_lib_output(${PROJECT_NAME}_Lib)


    ppr_blue("Build EXE: ${PROJECT_NAME}")
    add_executable(${PROJECT_NAME} main.cpp MUtil.h matrix_operations.cpp)


    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)

    include(CheckIPOSupported)
    check_ipo_supported(RESULT result)
    if (result)
        set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()

    target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_Lib)

    config_bin_output(${PROJECT_NAME})

endif ()