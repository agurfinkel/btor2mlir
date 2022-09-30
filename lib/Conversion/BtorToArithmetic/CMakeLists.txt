add_mlir_conversion_library(MLIRBtorToArithmetic
    BtorToArithmetic.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToArithmetic

    DEPENDS
    BTORConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )