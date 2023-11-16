add_mlir_dialect_library(MLIRBtor
        IR/BtorDialect.cpp
        IR/BtorOps.cpp
        IR/BtorTypes.cpp
        IR/BtorAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Btor

        DEPENDS
        MLIRBtorOpsIncGen
        BtorAttributesIncGen

	LINK_LIBS PUBLIC
    MLIRDialect
	MLIRIR
	)
