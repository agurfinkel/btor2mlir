add_mlir_dialect_library(MLIRBtorTranslate
    BtorToBtorIRTranslation.cpp
    BtorIRToBtorTranslation.cpp
    ${PROJECT_SOURCE_DIR}/include/Target/Btor/btor2parser/btor2parser.c

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/Btor

	LINK_LIBS PUBLIC
	MLIRIR
	)
