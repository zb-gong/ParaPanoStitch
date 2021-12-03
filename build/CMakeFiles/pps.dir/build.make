# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build

# Include any dependencies generated for this target.
include CMakeFiles/pps.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pps.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pps.dir/flags.make

CMakeFiles/pps.dir/src/main.cpp.o: CMakeFiles/pps.dir/flags.make
CMakeFiles/pps.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pps.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pps.dir/src/main.cpp.o -c /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/main.cpp

CMakeFiles/pps.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pps.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/main.cpp > CMakeFiles/pps.dir/src/main.cpp.i

CMakeFiles/pps.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pps.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/main.cpp -o CMakeFiles/pps.dir/src/main.cpp.s

CMakeFiles/pps.dir/src/util.cpp.o: CMakeFiles/pps.dir/flags.make
CMakeFiles/pps.dir/src/util.cpp.o: ../src/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pps.dir/src/util.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pps.dir/src/util.cpp.o -c /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/util.cpp

CMakeFiles/pps.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pps.dir/src/util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/util.cpp > CMakeFiles/pps.dir/src/util.cpp.i

CMakeFiles/pps.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pps.dir/src/util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/src/util.cpp -o CMakeFiles/pps.dir/src/util.cpp.s

# Object files for target pps
pps_OBJECTS = \
"CMakeFiles/pps.dir/src/main.cpp.o" \
"CMakeFiles/pps.dir/src/util.cpp.o"

# External object files for target pps
pps_EXTERNAL_OBJECTS =

pps: CMakeFiles/pps.dir/src/main.cpp.o
pps: CMakeFiles/pps.dir/src/util.cpp.o
pps: CMakeFiles/pps.dir/build.make
pps: /usr/local/lib/libopencv_stitching.so.3.4.16
pps: /usr/local/lib/libopencv_superres.so.3.4.16
pps: /usr/local/lib/libopencv_videostab.so.3.4.16
pps: /usr/local/lib/libopencv_aruco.so.3.4.16
pps: /usr/local/lib/libopencv_bgsegm.so.3.4.16
pps: /usr/local/lib/libopencv_bioinspired.so.3.4.16
pps: /usr/local/lib/libopencv_ccalib.so.3.4.16
pps: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.16
pps: /usr/local/lib/libopencv_dpm.so.3.4.16
pps: /usr/local/lib/libopencv_face.so.3.4.16
pps: /usr/local/lib/libopencv_freetype.so.3.4.16
pps: /usr/local/lib/libopencv_fuzzy.so.3.4.16
pps: /usr/local/lib/libopencv_hdf.so.3.4.16
pps: /usr/local/lib/libopencv_hfs.so.3.4.16
pps: /usr/local/lib/libopencv_img_hash.so.3.4.16
pps: /usr/local/lib/libopencv_line_descriptor.so.3.4.16
pps: /usr/local/lib/libopencv_optflow.so.3.4.16
pps: /usr/local/lib/libopencv_reg.so.3.4.16
pps: /usr/local/lib/libopencv_rgbd.so.3.4.16
pps: /usr/local/lib/libopencv_saliency.so.3.4.16
pps: /usr/local/lib/libopencv_stereo.so.3.4.16
pps: /usr/local/lib/libopencv_structured_light.so.3.4.16
pps: /usr/local/lib/libopencv_surface_matching.so.3.4.16
pps: /usr/local/lib/libopencv_tracking.so.3.4.16
pps: /usr/local/lib/libopencv_xfeatures2d.so.3.4.16
pps: /usr/local/lib/libopencv_ximgproc.so.3.4.16
pps: /usr/local/lib/libopencv_xobjdetect.so.3.4.16
pps: /usr/local/lib/libopencv_xphoto.so.3.4.16
pps: /usr/local/lib/libopencv_shape.so.3.4.16
pps: /usr/local/lib/libopencv_highgui.so.3.4.16
pps: /usr/local/lib/libopencv_videoio.so.3.4.16
pps: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.16
pps: /usr/local/lib/libopencv_video.so.3.4.16
pps: /usr/local/lib/libopencv_datasets.so.3.4.16
pps: /usr/local/lib/libopencv_plot.so.3.4.16
pps: /usr/local/lib/libopencv_text.so.3.4.16
pps: /usr/local/lib/libopencv_dnn.so.3.4.16
pps: /usr/local/lib/libopencv_ml.so.3.4.16
pps: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
pps: /usr/local/lib/libopencv_objdetect.so.3.4.16
pps: /usr/local/lib/libopencv_calib3d.so.3.4.16
pps: /usr/local/lib/libopencv_features2d.so.3.4.16
pps: /usr/local/lib/libopencv_flann.so.3.4.16
pps: /usr/local/lib/libopencv_photo.so.3.4.16
pps: /usr/local/lib/libopencv_imgproc.so.3.4.16
pps: /usr/local/lib/libopencv_core.so.3.4.16
pps: CMakeFiles/pps.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable pps"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pps.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pps.dir/build: pps

.PHONY : CMakeFiles/pps.dir/build

CMakeFiles/pps.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pps.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pps.dir/clean

CMakeFiles/pps.dir/depend:
	cd /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build /home/zibo/Class/15618_ParrallelComputing/ParaPanoStitch/build/CMakeFiles/pps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pps.dir/depend
