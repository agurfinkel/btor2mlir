add_mlir_conversion_library(MLIRBtorNDToLLVM
    BtorNDToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorNDToLLVM

    DEPENDS
    BTORConversionPassIncGen
    intrinsics_gen
    
    LINK_COMPONENTS
    Core
    
    LINK_LIBS PUBLIC
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    MLIRTransforms
    )