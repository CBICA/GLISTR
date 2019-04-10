
add_definitions(-DHAVE_ZLIB)

add_library(NIFTI nifti1_io.c znzlib.c)

if ( CMAKE_COMPILER_IS_GNUCC )
    # This is tested
    set_property( TARGET NIFTI APPEND_STRING PROPERTY COMPILE_FLAGS -w )
endif ( CMAKE_COMPILER_IS_GNUCC )
