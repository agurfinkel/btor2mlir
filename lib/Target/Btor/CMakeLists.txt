add_mlir_dialect_library(MLIRBtorTranslate
        BtorToBtorIRTranslation.cpp
        btor2tools

	LINK_LIBS PUBLIC
	MLIRIR
	)