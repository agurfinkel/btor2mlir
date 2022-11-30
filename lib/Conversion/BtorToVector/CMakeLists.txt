add_mlir_conversion_library(MLIRBtorToVector
    BtorToVector.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToVector

    DEPENDS
    BTORConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRVectorDialect
    )