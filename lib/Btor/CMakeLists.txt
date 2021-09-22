add_mlir_dialect_library(MLIRBtor
        BtorDialect.cpp
        BtorOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Btor

        DEPENDS
        MLIRBtorOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
