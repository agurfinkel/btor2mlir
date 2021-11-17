add_mlir_dialect_library(MLIRSmt
        IR/SmtDialect.cpp
        IR/SmtOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Smt

        DEPENDS
        MLIRSmtOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
