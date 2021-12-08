add_mlir_dialect_library(MLIRCrab
        IR/CrabDialect.cpp
        IR/CrabOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Crab

        DEPENDS
        MLIRCrabOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
