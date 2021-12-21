cmake_minimum_required(VERSION 3.12)
project(PyOpenVINS)

add_subdirectory(pybind11)
pybind11_add_module(PyOpenVINS bindings/ov_core.cpp
        )

find_package(Eigen3 REQUIRED)
include_directories(
        ${EIGEN3_INCLUDE_DIR}
)

find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 REQUIRED)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(../ov_core/src ../ov_msckf/src ../ov_eval/src ../ov_init/src)
set_property(TARGET PyOpenVINS PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(PyOpenVINS PUBLIC ov_core_lib ov_msckf_lib ov_eval_lib ov_init_lib)

install(TARGETS PyOpenVINS
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the exported targets
#install(
#        EXPORT PyOpenVINS
#        FILE PyOpenVINSTargets.cmake
#        NAMESPACE PyOpenVINS::
#        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyOpenVINS)


set(PyOpenVINS_BUILDDIR "${CMAKE_BINARY_DIR}/PyOpenVINS")
# For the python package we need an init file
file(
        GENERATE
        OUTPUT "${PyOpenVINS_BUILDDIR}/__init__.py"
        CONTENT "from PyOpenVINS.PyOpenVINS import *\n")

# Install the __init__.py file
install(
        FILES "${PyOpenVINS_BUILDDIR}/__init__.py"
        DESTINATION ${CMAKE_INSTALL_PREFIX})


