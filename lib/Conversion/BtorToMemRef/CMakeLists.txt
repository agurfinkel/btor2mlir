add_mlir_conversion_library(MLIRBtorToMemRef
    BtorToMemRef.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToMemRef

    DEPENDS
    BTORConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )