cmake_minimum_required (VERSION 2.6)
project (Tutorial)
# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)
 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "/home/chris/Documents/MLProject/cmake-fiddle/TutorialConfig.h.in"
  "/home/chris/Documents/MLProject/cmake-fiddle/TutorialConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("/home/chris/Documents/MLProject/cmake-fiddle")
 
# add the executable
add_executable(Tutorial tutorial.cxx)

include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
add_subdirectory (MathFunctions) 
 
# add the executable
add_executable (Tutorial tutorial.cxx)
target_link_libraries (Tutorial MathFunctions)