add_library(zlib adler32.c compress.c crc32.c deflate.c gzclose.c gzlib.c gzread.c gzwrite.c infback.c inffast.c inflate.c inftrees.c trees.c uncompr.c zutil.c)

if ( CMAKE_COMPILER_IS_GNUCC )
    # This is tested
    set_property( TARGET zlib APPEND_STRING PROPERTY COMPILE_FLAGS -w )
endif ( CMAKE_COMPILER_IS_GNUCC )
