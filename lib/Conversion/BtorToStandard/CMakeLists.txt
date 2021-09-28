add_mlir_conversion_library(MLIRBtorToStandard
    BtorToStandard.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToStandard

    DEPENDS
    BTORConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )