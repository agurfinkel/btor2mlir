add_mlir_conversion_library(MLIRBtorToMath
    BtorToMath.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToMath

    DEPENDS
    BTORConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )