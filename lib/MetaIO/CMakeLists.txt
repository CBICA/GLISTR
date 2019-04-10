add_library(MetaIO metaArray.cxx metaArrow.cxx metaBlob.cxx metaCommand.cxx metaContour.cxx metaDTITube.cxx metaEllipse.cxx metaForm.cxx metaGaussian.cxx metaGroup.cxx metaImage.cxx metaImageUtils.cxx metaLandmark.cxx metaLine.cxx metaMesh.cxx metaObject.cxx metaOutput.cxx metaScene.cxx metaSurface.cxx metaTransform.cxx metaTube.cxx metaTubeGraph.cxx metaUtils.cxx metaVesselTube.cxx)

if ( CMAKE_COMPILER_IS_GNUCC )
    # This is tested
    set_property( TARGET MetaIO APPEND_STRING PROPERTY COMPILE_FLAGS -w )
endif ( CMAKE_COMPILER_IS_GNUCC )
